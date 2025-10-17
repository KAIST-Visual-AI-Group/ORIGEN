import os
import argparse
from dataclasses import dataclass
from PIL import ImageDraw
from tqdm import tqdm
import json
import torchvision

from src.utils import *
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

    if args.data_path.endswith(".json"):
        dataset = json.load(open(args.data_path, 'r'))
    else:
        with open(args.data_path, 'r') as f:
            dataset = [line.strip() for line in f if line.strip() != '']
    
    for idx, data in enumerate(tqdm(dataset, total=len(dataset), desc="Benchmark")):
        # which mode
        if "multi" in CFG.mode:
            orientations = [[bboxes[0] for bboxes in data['orientations'][0]]]
        else:
            orientations = data['orientations'][0]
            
        for orient_idx in range(len(orientations)):
            data['orientations'] = orientations[orient_idx]
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
            image.save(os.path.join(args.save_dir, f"prompt_{idx}.png"))
            
            if args.save_reward:
                draw = ImageDraw.Draw(image)
                text = f"{best_reward.item():.5f}" if hasattr(best_reward, "item") else f"{best_reward:.5f}"
                draw.rectangle([0, 0, 60, 20], fill=(0, 0, 0, 128))  
                draw.text((5, 2), text, fill=(255, 255, 255))
                
                # draw angle
                estimated_angles = reward_model.get_angle(best_sample)
                estimated_bboxes = reward_model.estimated_bboxes
                
                image = draw_orientation(image, estimated_bboxes, estimated_angles)
                image.save(os.path.join(args.save_dir, f"prompt_{idx}_orientation_rendered.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/orientation_grounding.yaml")
    parser.add_argument("--data_path", default="./data/ORIBENCH_single.json")
    parser.add_argument("--save_dir", default="./results_bench")
    parser.add_argument("--save_reward", action="store_true", help="Save the reward value on the image")

    args, extras = parser.parse_known_args()
    CFG = load_config(args.config, cli_args=extras)

    args.save_dir = f"{args.save_dir}_{CFG.mode}"
    CFG.save_dir = args.save_dir
    CFG.mode = args.data_path.split("/")[-1].split(".")[0].split("_")[-1]
    if args.save_reward:
        os.makedirs(args.save_dir, exist_ok=True)

    main(CFG, args)