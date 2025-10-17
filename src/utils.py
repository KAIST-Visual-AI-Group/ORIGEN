import os
import sys
from dataclasses import fields
import random
import math
from render import render, Model

import numpy as np
import torch

from omegaconf import OmegaConf, DictConfig
from PIL import Image
from src.orientation_grounding.orient_utils import find_phrase_idx


class StrictStdoutSuppressor:
    def __init__(self, allowed_prefix="[MYPRINT]"):
        self.allowed_prefix = allowed_prefix
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def write(self, msg):
        if msg.startswith(self.allowed_prefix):
            self.original_stdout.write(msg[len(self.allowed_prefix):])
        return

    def flush(self):
        self.original_stdout.flush()
        self.original_stderr.flush()

def print_my(*args, **kwargs):
    msg = ' '.join(str(arg) for arg in args)
    sys.__stdout__.write(msg + '\n')  
    sys.stdout.write("[MYPRINT]" + msg + '\n')  

def suppress_print():
    return StrictStdoutSuppressor()

def ignore_kwargs(cls):
    original_init = cls.__init__

    def init(self, *args, **kwargs):
        expected_fields = {field.name for field in fields(cls)}
        expected_kwargs = {
            key: value for key, value in kwargs.items() if key in expected_fields
        }
        original_init(self, *args, **expected_kwargs)

    cls.__init__ = init
    return cls

def load_config(*yamls, cli_args = None, from_string=False, **kwargs):
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    return cfg

def seed_everything(seed, use_deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tensor2PIL(tensor, do_normalize=False):
    tensor = tensor.clone().detach().cpu().float()
    if do_normalize:
        tensor = tensor / 2.0 + 0.5
    tensor = tensor.clip(0, 1).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    tensor = np.transpose(tensor, (1, 2, 0))
    image = Image.fromarray(tensor)
    return image
    
    
def calculate_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    
    y_center = int((y_max + y_min)/2)
    x_center = int((x_max + x_min)/2)
    return (y_center, x_center)


axis_model = Model("./assets/axis.obj", texture_filename="./assets/axis.png")
def render_3D_axis(phi, theta, gamma):
    radius = 240
    # camera_location = [radius * math.cos(phi), radius * math.sin(phi), radius * math.tan(theta)]
    camera_location = [-1*radius * math.cos(phi), -1*radius * math.tan(theta), radius * math.sin(phi)]
    img = render(
        # Model("res/jinx.obj", texture_filename="res/jinx.tga"),
        axis_model,
        height=512,
        width=512,
        filename="tmp_render.png",
        cam_loc = camera_location
    )
    img = img.rotate(gamma)
    return img

def overlay_images_with_scaling(center_image: Image.Image, background_image: Image.Image, center=None, target_size=(512, 512)):
    """
    :param center_image: Image to overlay (rendered axis)
    :param background_image: Background image
    :param center: (x, y) coordinates for placing the center image
    :param target_size: Target size for the result (default: 512x512)
    :return: Final image with the center image overlaid on the background
    """
    
    if center_image.mode != "RGBA":
        center_image = center_image.convert("RGBA")
    if background_image.mode != "RGBA":
        background_image = background_image.convert("RGBA")

    # Background size is fixed to (512, 512)
    bg_width, bg_height = background_image.size
    
    # Resize the center image to be smaller if needed, ensuring it doesn't exceed the background size
    max_width = int(bg_width * 0.5)  # Example: center image should be at most 50% of background width
    max_height = int(bg_height * 0.5)  # Example: center image should be at most 50% of background height

    center_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)  # Resize while maintaining aspect ratio

    # Create a copy of the background image to overlay the center image
    result = background_image.copy()

    # Calculate the position to overlay the center image at the specified center
    if center is not None:
        center_y, center_x = center
    else:
        center_y, center_x = bg_height // 2, bg_width // 2
        
    position_x = int(center_x - center_image.width // 2)
    position_y = int(center_y - center_image.height // 2)

    # Make sure the position is within the bounds of the background image
    position_x = max(0, min(position_x, result.width - center_image.width))
    position_y = max(0, min(position_y, result.height - center_image.height))

    # Overlay the center image at the calculated position
    result.paste(center_image, (position_x, position_y), mask=center_image)

    return result

      
def draw_orientation(img, estimated_bboxes, estimated_orientations):
    for bbox, orientation in zip(estimated_bboxes, estimated_orientations):
        azimuth, polar, rotation = orientation[0].cpu().numpy(), orientation[1].cpu().numpy(), orientation[2].cpu().numpy()
        angles = [azimuth, polar - 90, rotation - 90]
        
        center = calculate_center(bbox)
        
        azimuth = float(np.radians(angles[0]))
        polar = float(np.radians(angles[1]))
        rotation = float(np.radians(angles[2]))
        render_axis = render_3D_axis(azimuth, polar, rotation)
        img = overlay_images_with_scaling(render_axis, img, center=center)
    return img

def preprocess_prompt(prompt, phrases, orientations):
    for phrase, obj_orientation in zip(phrases, orientations):
        phrase_end_idx_in_prompts = find_phrase_idx(prompt, phrase)
        
        flag = False
        azimuth_angle = obj_orientation[0]
        if 130 < azimuth_angle < 230:
            flag = True
        
        if flag:
            prompt = prompt[:phrase_end_idx_in_prompts] + ", back view" + prompt[phrase_end_idx_in_prompts:]
    return prompt