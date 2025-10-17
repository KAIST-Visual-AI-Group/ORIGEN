import os
import argparse
from dataclasses import dataclass
from PIL import ImageDraw
import json
import torch
import torchvision

from src.utils import seed_everything, load_config, suppress_print, preprocess_prompt, draw_orientation, ignore_kwargs
from src.flux_pipeline import FluxSchnellPipeline
from src.reward_model import get_reward_model


@ignore_kwargs
@dataclass
class Config:
    seed: int = 0
    negative_prompt: str = None
    height: int = 512
    width: int = 512

def main(CFG, args):
    device = torch.device("cuda:0")
    task_name = args.config.split("/")[-1].split(".")[0]
    seed_everything(CFG.seed)

    with suppress_print():
        pipe = FluxSchnellPipeline(device, CFG)
        reward_model = get_reward_model(task_name)(torch.float32, device, CFG)
    
    # load data
    data = json.load(open(args.data_path, 'r'))

    orientations = data['orientations'][0][0]
    data['orientations'] = orientations
    prompt = data["prompts"]
    phrases = data['phrases']
    # prompt preprocessing
    prompt = preprocess_prompt(prompt, phrases, orientations)

    pipe.load_encoder()
    pipe.encode_prompt(prompt, CFG.negative_prompt, phrases=phrases)
    reward_model.register_data(data)
    pipe.unload_encoder()

    generator = torch.Generator(device=device).manual_seed(CFG.seed)
    
    _, best_sample, best_reward = pipe.sample(height=CFG.height, width=CFG.width, reward_model=reward_model, generator=generator)
        
    image = torchvision.transforms.ToPILImage()(best_sample[0].float().cpu().clamp(0, 1))
    image.save(os.path.join(args.save_dir, f"output.png"))
    
    if args.save_reward:
        draw = ImageDraw.Draw(image)
        text = f"{best_reward.item():.5f}" if hasattr(best_reward, "item") else f"{best_reward:.5f}"
        draw.rectangle([0, 0, 60, 20], fill=(0, 0, 0, 128))  
        draw.text((5, 2), text, fill=(255, 255, 255))
        
        # draw angle
        estimated_angles = reward_model.get_angle(best_sample)
        estimated_bboxes = reward_model.estimated_bboxes
        
        image = draw_orientation(image, estimated_bboxes, estimated_angles)
        image.save(os.path.join(args.save_dir, f"output_orientation_rendered.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/orientation_grounding.yaml")
    parser.add_argument("--data_path", default="./data/single.json")
    parser.add_argument("--save_reward", action="store_true")
    parser.add_argument("--save_dir", default="./outputs")

    args, extras = parser.parse_known_args()
    CFG = load_config(args.config, cli_args=extras)

    CFG.save_dir = args.save_dir
    if args.save_reward:
        os.makedirs(args.save_dir, exist_ok=True)

    main(CFG, args)