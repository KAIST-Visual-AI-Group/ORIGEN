import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from dataclasses import dataclass
from src.utils import ignore_kwargs

# Orientation Grounding Reward
from src.orientation_grounding.orient_anything import DINOv2_MLP
from src.orientation_grounding.orient_utils import estimate_bboxes, differentiable_background_preprocess, kl


__rewards__ = dict()

def register_reward_model(name):
    def decorator(cls):
        __rewards__[name] = cls
        return cls
    return decorator

def get_reward_model(name: str):
    if name not in __rewards__:
        raise ValueError(f"Reward model {name} not found. Available reward models: {list(__rewards__.keys())}")
    return __rewards__[name]


@register_reward_model(name="orientation_grounding")
class OrientationGroundingReward(nn.Module):
    @ignore_kwargs
    @dataclass
    class Config():
        decode_to_unnormalized: bool = True
        grad_clip: float = None
        grad_norm: float = None
        
        azimuth_only: bool = False,
        polar_only: bool = False,

        scaling_factor: float = 0.85,
        
        save_vram : bool = True
        early_stop : bool = True

    def __init__(self, dtype, device, CFG):
        super().__init__()
        self.cfg = self.Config(**CFG)
        self.dtype = dtype
        self.device = device
        
        ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="croplargeEX2/dino_weight.pt", repo_type="model", resume_download=True)
        
        orient_estimator = DINOv2_MLP(
            dino_mode='large',
            in_dim=1024,
            out_dim=360+180+180+2,
            evaluate=True,
            mask_dino=False,
            frozen_back=False,
            dtype=dtype
        )
        orient_estimator.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        
        # orient estimator mean, std
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype).view(3, 1, 1).to(self.device)
        self.orient_estimator = orient_estimator.to(device=device, dtype=dtype)
        
        # Load the processor
        bboxmaker_id = "IDEA-Research/grounding-dino-base"
        
        self.gd_processor = AutoProcessor.from_pretrained(bboxmaker_id)
        self.object_detector = AutoModelForZeroShotObjectDetection.from_pretrained(bboxmaker_id).to(self.device)
        
        # Cache given information
        self.azimuth_only = self.cfg.azimuth_only
        self.polar_only = self.cfg.polar_only
        self.scaling_factor = self.cfg.scaling_factor
        self.estimated_bboxes = None
        
        # Save VRAM
        if self.cfg.save_vram:
            self.orient_estimator.eval().requires_grad_(False)
            self.object_detector.eval().requires_grad_(False)
            
        if self.cfg.early_stop:
            self.success = False
            self.success_sample = None

    def preprocess_tensor(self, x):
        """
        Preprocess the input image tensor.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        # 1. Resize
        x = x.to(self.dtype)
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        
        # 2. CenterCrop
        x = TF.center_crop(x, (224, 224))
        
        # 3. Normalize
        x = (x - self.mean) / self.std
        
        # 4. Return back to original dtype
        x = x.to(self.dtype)
        return x

    def make_target_distributions(self, input_target_orientations):
        """
        Generate Gaussian distributions for azimuth, polar, and rotation for each target orientation.
        Args:
            target_orientations (list): A list of target orientations for each object.
        Returns:
            torch.Tensor: A tensor of shape (N, 360+180+180) containing the target distributions.
        """

        # NOTE: For now, support only one object for each object category

        target_orientations = torch.tensor(input_target_orientations, dtype=self.dtype).to(self.device)
        azimuth_distribution, polar_distribution, rotation_distribution = self.orientations_to_distribution(target_orientations)
        target_distributions = torch.cat([azimuth_distribution, polar_distribution, rotation_distribution], dim=1)
        return target_distributions

    def orientations_to_distribution(self, orientations, azimuth_sigma=20.0):
        """
        Generate Gaussian distributions for azimuth, polar, and rotation for each orientation in batch.

        Args:
            orientations (torch.Tensor): A tensor of shape (N, 3), where each row represents an orientation (azimuth, polar, rotation).
            azimuth_sigma (float): Standard deviation for the azimuth Gaussian distribution.

        Returns:
            tuple: A tuple containing azimuth, polar, and rotation distributions for each orientation in the batch.
        """
            
        orientations = orientations.to(self.device)
        azimuths, polars, rotations = orientations[:, 0], orientations[:, 1], orientations[:, 2]

        # Polar distribution
        polar_sigma = 2.0
        polar_range = 180
        angles = torch.arange(1, polar_range + 1, dtype=self.dtype).to(self.device) 
        polar_distribution = torch.exp(-((angles - polars.unsqueeze(1)) ** 2) / (2 * polar_sigma ** 2))
        polar_distribution = polar_distribution / torch.sum(polar_distribution, dim=1, keepdim=True) 

        # Azimuth distribution (circular distance)
        azimuth_range = 360
        angles = torch.arange(1, azimuth_range + 1, dtype=self.dtype).to(self.device)
        circular_distance = torch.min(torch.abs(angles - azimuths.unsqueeze(1)), azimuth_range - torch.abs(angles - azimuths.unsqueeze(1)))
        azimuth_distribution = torch.exp(-0.5 * (circular_distance / azimuth_sigma) ** 2)
        azimuth_distribution /= torch.sum(azimuth_distribution, dim=1, keepdim=True)

        # Rotation distribution
        rotation_sigma = 1.0
        rotation_range = 180
        angles = torch.arange(1, rotation_range + 1, dtype=self.dtype).to(self.device) 
        rotation_distribution = torch.exp(-((angles - rotations.unsqueeze(1)) ** 2) / (2 * rotation_sigma ** 2))
        rotation_distribution = rotation_distribution / torch.sum(rotation_distribution, dim=1, keepdim=True)

        return azimuth_distribution, polar_distribution, rotation_distribution

    def predict_orientation(self, x):
        """
        Compute the orientation reward for a batch of images.

        Args:
            x (torch.Tensor): Input Tensor (1, C, H, W).

        Returns:
            List[List[torch.Tensor]]: Estimated Orientations. [[(M, 360+180+180), ...], ...]
        """
        assert isinstance(x, torch.Tensor), "Input x must be a torch tensor"
        assert x.ndim == 4 and x.size(0) == 1, "Input x must have following shape (1, C, H, W)"
        x = x.to(self.dtype).squeeze(0)  # (C, H, W)
        
        # Detect objects in image
        estimated_bboxes = estimate_bboxes(self.object_detector, self.gd_processor, x, self.phrases_list, self.predict_object_nums_list, device=self.device)
        
        # Background preprocess with differentiable manner
        # We crop different sizes for each image, so we need to process them separately
        processed_obj_images = differentiable_background_preprocess(x, estimated_bboxes, self.scaling_factor)

        all_preprocessed_imgs = []
        all_processed_bboxes = []
        for obj_imgs, obj_bboxes in zip(processed_obj_images, estimated_bboxes):
            for obj_img, obj_bbox in zip(obj_imgs, obj_bboxes):
                all_preprocessed_imgs.append(self.preprocess_tensor(obj_img))
                all_processed_bboxes.append(torch.tensor([obj_bbox], dtype=self.dtype).to(self.device))
        
        input_tensors = torch.cat(all_preprocessed_imgs, dim=0)
        
        # cache estimated bboxes
        self.estimated_bboxes = torch.cat(all_processed_bboxes, dim=0)
        
        # Get estimated orientations
        estimated = self.orient_estimator.inference(input_tensors)  # (N, 360+180+180+2)
        estimated_orientations = estimated[:, :-2]
        
        return estimated_orientations
    
    def get_angle(self, x):
        estimated_orientations = self.predict_orientation(x)
        
        azimuth_est = torch.argmax(estimated_orientations[:, :360], dim=1)
        polar_est = torch.argmax(estimated_orientations[:, 360:540], dim=1)
        rotation_est = torch.argmax(estimated_orientations[:, 540:720], dim=1)
        
        angle_output = torch.cat([azimuth_est.unsqueeze(1), polar_est.unsqueeze(1), rotation_est.unsqueeze(1)], dim=1)
        return angle_output
    
    def check_success(self, estimated_orientations):
        azimuth_est = torch.argmax(estimated_orientations[:, :360], dim=1)
        polar_est = torch.argmax(estimated_orientations[:, 360:540], dim=1)
        rotation_est = torch.argmax(estimated_orientations[:, 540:720], dim=1)
        # check if all angles are in range of 22.5
        tolerance = 22.5
        
        azimuth_diff = torch.min(torch.abs(azimuth_est - self.target_orientations[:, 0]), 360 - torch.abs(azimuth_est - self.target_orientations[:, 0])) <= tolerance # (N,)
        polar_diff = torch.abs(polar_est - self.target_orientations[:, 1]) <= tolerance  # (N,)
        rotation_diff = torch.abs(rotation_est - self.target_orientations[:, 2]) <= tolerance  # (N,)
        
        success = azimuth_diff & polar_diff & rotation_diff  # (N,)
        return all(success)
        
        
    def get_estimated_bboxes(self):
        return self.estimated_bboxes

    def compute_reward(self, x):
        reward = 0

        estimated_orientations = self.predict_orientation(x)
        target_distributions = self.target_distributions
            
        if self.azimuth_only:
            reward += kl(target_distributions[:, :360], estimated_orientations[:, :360])
        elif self.polar_only:
            reward += kl(target_distributions[:, 360:540], estimated_orientations[:, 360:540])
        else:
            reward += 1 * kl(target_distributions[:, :360], estimated_orientations[:, :360])
            reward += 1 * kl(target_distributions[:, 360:540], estimated_orientations[:, 360:540])
            reward += 1 * kl(target_distributions[:, 540:720], estimated_orientations[:, 540:720])
        
        reward = -reward.mean().unsqueeze(0)

        if self.cfg.early_stop:
            success = self.check_success(estimated_orientations)
            if success:
                self.success = True
                self.success_sample = x
                
        return reward
    
    def register_data(self, data):
        target_orientations = data['orientations']
        phrases = data['phrases']
        
        self.target_orientations = torch.tensor(target_orientations, dtype=self.dtype).to(self.device)
        self.target_distributions = self.make_target_distributions(target_orientations)
        self.phrases_list = phrases
        self.predict_object_nums_list = [1 for _ in range(len(phrases))]

    def __call__(self, x, _):
        return self.compute_reward(x)