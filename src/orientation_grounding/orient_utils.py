import numpy as np
import torch
import torchvision.transforms as T

from src.orientation_grounding.render_utils import render, Model
import math

from PIL import Image


def estimate_bboxes(object_detector, gd_processor, image, phrases, predict_object_nums_list=None, device=None):
    """
    Args:
        image: PIL.Image or torch.Tensor
        phrases: List[str], shared across all images
        predict_object_nums_list: List[int], shared across all images
        device: torch.device
    Returns:
        Dict[int, List[List[float]]], where Dict[label_id] = list of bboxes
    """
    
    if isinstance(image, torch.Tensor):
        # Convert to PIL Image
        image = T.ToPILImage()(image.detach().clone().cpu().to(torch.float32))
        
    text = ". ".join(phrases) + "."

    gd_inputs = gd_processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = object_detector(**gd_inputs)

    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        gd_inputs.input_ids,
        box_threshold=0.2,
        text_threshold=0.2,
        target_sizes=[image.size[::-1]]
    )[0]

    gt_labels = {label.replace(" ", ""): i for i, label in enumerate(text.split(".")[:-1])}

    results["labels"] = torch.tensor([gt_labels.get(label, -1) for label in results["labels"]])
    indices = torch.argwhere(results["labels"] >= 0).reshape(-1)

    for key in results:
        if isinstance(results[key], torch.Tensor):
            results[key] = results[key][indices].cpu()

    all_labels = list(gt_labels.values()) 
    unique_labels = set(results["labels"].tolist()) 

    image_width, image_height = image.size 
    selected_boxes = {}

    for label, required_num in zip(all_labels, predict_object_nums_list):
        if label in unique_labels:
            indices = torch.where(results["labels"] == label)[0] 
            sorted_indices = indices[torch.argsort(results["scores"][indices], descending=True)] 
            selected_bboxes = results["boxes"][sorted_indices].tolist() 
            
            if len(selected_bboxes) > required_num:
                selected_bboxes = selected_bboxes[:required_num]

            while len(selected_bboxes) < required_num:
                selected_bboxes.append(selected_bboxes[0])
                
        else:
            full_image_bbox = [0, 0, image_width, image_height]
            selected_bboxes = [full_image_bbox] * required_num  

        selected_boxes[label] = selected_bboxes 

    bboxes = [bbox for _, bbox in selected_boxes.items()]
    return bboxes



def resize_foreground_torch(
    image: torch.Tensor,
    nonzero_coords: torch.Tensor
) -> torch.Tensor:
    
    assert image.ndim == 3 and image.shape[0] == 3
    if nonzero_coords[0].numel() == 0:
        y2 = image.shape[1] 
        y1 = 0
    else:
        y1, y2 = nonzero_coords[0].min().item(), nonzero_coords[0].max().item()
    if nonzero_coords[1].numel() == 0:
        x2 = image.shape[2] 
        x1 = 0
    else:
        x1, x2 = nonzero_coords[1].min().item(), nonzero_coords[1].max().item()

    # Crop the foreground (keeping all channels)
    fg = image[:, y1:y2, x1:x2] 
    return fg


# Background Preprocessing with differentiable manner
def differentiable_background_preprocess(decoded_x0, estimated_bboxes, scaling_factor=0.85) -> torch.Tensor:
    # Preprocess background with differentiable manner
    """
    Input:
        decoded_x0: torch.Tensor,  (C, H, W)
    """
    assert isinstance(decoded_x0, torch.Tensor), "decoded_x0 must be a torch tensor"
    assert decoded_x0.ndim == 3, "decoded_x0 must have 3 dimensions (C, H, W)"
    
    C, H, W = decoded_x0.shape

    # postprocess decoded_x0
    # Expand bounding box
    decoded_x0s = []
    
    for obj_bboxes in estimated_bboxes:
        decoded_obj_x0s = []
        for bbox in obj_bboxes:
            x_min, y_min, x_max, y_max = bbox
            
            scale_factor = 1 / scaling_factor
            box_height = y_max - y_min + 1
            box_width = x_max - x_min + 1

            new_y_min = max(0, int(y_min - (scale_factor - 1) * box_height / 2))
            new_y_max = min(H, int(y_max + (scale_factor - 1) * box_height / 2))
            new_x_min = max(0, int(x_min - (scale_factor - 1) * box_width / 2))
            new_x_max = min(W, int(x_max + (scale_factor - 1) * box_width / 2))
            
            # Create expanded bounding box mask
            bounding_box_mask = torch.zeros_like(decoded_x0).to(decoded_x0.dtype).to(decoded_x0.device)
            bounding_box_mask[:, new_y_min:new_y_max, new_x_min:new_x_max] = 1.0
            
            # Apply mask
            removed_x0 = bounding_box_mask * decoded_x0
            
            # Resize foreground object
            nonzero_coords = (torch.tensor([new_y_min, new_y_max]), torch.tensor([new_x_min, new_x_max]))
            removed_x0 = resize_foreground_torch(removed_x0, nonzero_coords=nonzero_coords)
            
            decoded_obj_x0s.append(removed_x0.unsqueeze(0))
        
        decoded_x0s.append(decoded_obj_x0s)

    return decoded_x0s


def kl(y, pred):
    return torch.nn.functional.kl_div(pred.to(torch.float32).log(), y.to(torch.float32), reduction='batchmean')



# Rendering utils
axis_model = Model("assets/axis.obj", texture_filename="assets/axis.png")

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

def calculate_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    
    y_center = int((y_max + y_min)/2)
    x_center = int((x_max + x_min)/2)
    return (y_center, x_center)

def render_3D_axis_on_image(estimated_orientations, estimated_bboxes, image, target_size=(512, 512)):
    """
    Args:
        estimated_orientations: torch.Tensor of shape (N, 360+180+180)
        estimated_bboxes: torch.Tensor of shape (N, 4)
        image: PIL.Image
        target_size: tuple of target size for the result image
    Returns:
        PIL.Image with the rendered axis overlaid on the image
    """
    res_img = image.copy()
    for obj_orientation, obj_bbox in zip(estimated_orientations, estimated_bboxes):
        azimuth = torch.argmax(obj_orientation[:360], dim=-1).item()
        polar = torch.argmax(obj_orientation[360:540], dim=-1).item()
        rotation = torch.argmax(obj_orientation[540:720], dim=-1).item()

        angles = [azimuth, polar - 90, rotation - 90]
        center = calculate_center(obj_bbox)
        
        azimuth = float(np.radians(angles[0]))
        polar = float(np.radians(angles[1]))
        rotation = float(angles[2])
        
        # Rendering axis
        render_axis = render_3D_axis(azimuth, polar, rotation)

        # Overlay the axis image on the background image
        res_img = overlay_images_with_scaling(render_axis, res_img, center, target_size)

    return res_img

def find_phrase_idx(prompt, phrase):
    start_index = prompt.find(phrase)
    
    if start_index != -1:
        end_index = start_index + len(phrase) - 1
    else:
        end_index = -1
    
    return end_index + 1