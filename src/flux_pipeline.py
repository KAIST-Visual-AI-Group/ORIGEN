from dataclasses import dataclass

import torch

from src.utils import ignore_kwargs
from src.scheduler import get_scheduler
from src.method import get_sampler
from diffusers import FluxPipeline


class TimeSampler():
    @ignore_kwargs
    @dataclass
    class Config:
        num_inference_steps: int = 1
        time_schedule: str = "linear"

    def __init__(self, device, CFG):
        self.cfg = self.Config(**CFG)

        if self.cfg.time_schedule == "linear":
            self.times = torch.linspace(1.0, 1.0 / self.cfg.num_inference_steps, self.cfg.num_inference_steps, device=device)
        elif self.cfg.time_schedule == "nonlinear":
            x = torch.linspace(0.0, 1.0 - 1 / self.cfg.num_inference_steps, self.cfg.num_inference_steps, device=device)
            self.times = (1 - x ** 2) ** 0.5
        else:
            raise ValueError(f"Unknown time schedule {self.cfg.time_schedule}. Use 'linear' or 'nonlinear'.")
        
        self.times = torch.cat([self.times, torch.zeros(1, device=self.times.device)]).to(torch.float32)

    def __call__(self, step):
        if type(step) not in [torch.Tensor]:
            if type(step) in [int, float]:
                step = [step]
            step = torch.tensor(step, device=self.times.device)
        assert (step < self.cfg.num_inference_steps).all().item(), f"step {step} >= num inference step {self.cfg.num_inference_steps}"
        return self.times[step], self.times[step + 1]


class FluxSchnellPipeline():
    @ignore_kwargs
    @dataclass
    class Config:
        model: str = "schnell"
        batch_size: int = 1  # Support only batch_size = 1 for now

        true_cfg_scale: float = 1.0
        guidance_scale: float = 3.5  # Schnell does not use this guidance_scale (Dev does)

        original_scheduler: str = "linear"

        method: str = "origen"  # Sampling method

        save_vram : bool = False

    def __init__(self, device, CFG):
        self.cfg = self.Config(**CFG)
        self.device = device
    
        model_id = "black-forest-labs/FLUX.1-schnell"
        
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

        # Save VRAM
        if self.cfg.save_vram:
            self.pipe.vae.eval().requires_grad_(False)
            self.pipe.text_encoder.eval().requires_grad_(False)
            self.pipe.text_encoder_2.eval().requires_grad_(False)
            self.pipe.transformer.eval().requires_grad_(False)

        self.original_scheduler = get_scheduler(self.cfg.original_scheduler)()

        self.guidance = None

        # Initial Latents Sampling Method
        if self.cfg.method is not None:
            self.init_sampling_method = get_sampler(self.cfg.method)(CFG)
        else:
            self.init_sampling_method = None

        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.height = None
        self.width = None

    def unload_encoder(self):
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")

    def load_encoder(self):
        self.pipe.text_encoder.to(self.device)
        self.pipe.text_encoder_2.to(self.device)

    def clear_cache(self):
        self.prompt_embeds = None
        self.pooled_peompt_embeds = None

        if self.negative_prompt_embeds is not None:
            self.negative_prompt_embeds = None
            self.negative_pooled_prompt_embeds = None

        self.text_ids = None
        self.latent_image_ids = None
        self.negative_prompt_embeds = None
        self.height = None
        self.width = None
    
    def encode_prompt(self, prompt, negative_prompt=None, prompt_2=None, negative_prompt_2=None, phrases=None):
        self.do_true_cfg = self.cfg.true_cfg_scale > 1.0 and negative_prompt is not None

        self.prompt_embeds, self.pooled_peompt_embeds, self.text_ids = self.pipe.encode_prompt(
            prompt = prompt,
            prompt_2 = prompt_2,
            device = self.device)

        if self.do_true_cfg:
            self.negative_prompt_embeds, self.negative_pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt = negative_prompt,
                prompt_2 = negative_prompt_2,
                device = self.device)
        
        if phrases is not None:
            self.phrases_indices = self.get_t5_subsequence_indices(prompt, phrases)

    def get_t5_subsequence_indices(self, prompt, phrases_list):
        tokens = self.pipe.tokenizer_2.encode(prompt)
        
        phrases_indices = []
        for phrase in phrases_list:
            sub_tokens = self.pipe.tokenizer_2.encode(phrase)[:-1]
            # Find subsequence input_ids
            sub_len = len(sub_tokens)
            for i in range(len(tokens) - sub_len + 1):
                if tokens[i:i + sub_len] == sub_tokens:
                    phrases_indices.append([j for j in range(i, i + sub_len)])
        
        return phrases_indices

    def sample(self, height, width, reward_model=None, generator=None):
        self.height = height
        self.width = width
        self.latent_h, self.latent_w = int(height) // (self.pipe.vae_scale_factor * 2), int(width) // (self.pipe.vae_scale_factor * 2)
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        batch_size = self.cfg.batch_size

        latents, latent_image_ids = self.pipe.prepare_latents(
            batch_size = batch_size,
            num_channels_latents = num_channels_latents,
            height = height,
            width = width,
            dtype = self.pipe.dtype,
            device = self.device,
            generator = generator)
        self.latent_image_ids = latent_image_ids
        
        samples, best_sample, best_reward = self.init_sampling_method(latents, reward_model=reward_model, pipe=self)
        return samples, best_sample, best_reward
    
    def predict(self, latents, t):
        vel_pred = list()
        for i in range(0, latents.shape[0], self.cfg.batch_size):
            cur_batch_size = min(self.cfg.batch_size, latents.shape[0] - i)
            cur_latents = latents[i:i+cur_batch_size]
            cur_t = t[i:i+cur_batch_size].to(latents.dtype)

            cur_guidance = self.guidance.expand(cur_latents.shape[0]) if self.guidance is not None else None
            cur_pooled_prompt_embeds = self.pooled_peompt_embeds.repeat(cur_batch_size, *([1] * (self.pooled_peompt_embeds.dim() - 1)))
            cur_prompt_embeds = self.prompt_embeds.repeat(cur_batch_size, *([1] * (self.prompt_embeds.dim() - 1)))

            cur_vel_pred = self.pipe.transformer(
                hidden_states = cur_latents,
                timestep = cur_t,
                guidance = cur_guidance,
                pooled_projections = cur_pooled_prompt_embeds,
                encoder_hidden_states = cur_prompt_embeds,
                txt_ids = self.text_ids,
                img_ids = self.latent_image_ids,
                joint_attention_kwargs = {},
                return_dict = False)[0]

            if self.do_true_cfg:
                assert self.negative_prompt_embeds is not None, "Negative prompt embeddings must be encoded first."

                cur_neg_pooled_prompt_embeds = self.negative_pooled_prompt_embeds.repeat(cur_batch_size, *([1] * (self.negative_pooled_prompt_embeds.dim() - 1)))
                cur_neg_prompt_embeds = self.negative_prompt_embeds.repeat(cur_batch_size, *([1] * (self.negative_prompt_embeds.dim() - 1)))

                cur_neg_vel_pred = self.pipe.transformer(
                    hidden_states = cur_latents,
                    timestep = cur_t,
                    guidance = cur_guidance,
                    pooled_projections = cur_neg_pooled_prompt_embeds,
                    encoder_hidden_states = cur_neg_prompt_embeds,
                    txt_ids = self.text_ids,
                    img_ids = self.latent_image_ids,
                    joint_attention_kwargs = {},
                    return_dict = False)[0]
                cur_vel_pred = cur_neg_vel_pred + self.cfg.true_cfg_scale * (cur_vel_pred - cur_neg_vel_pred)
            vel_pred.append(cur_vel_pred)

        vel_pred = torch.cat(vel_pred, dim=0)
        return vel_pred
    
    def forward(self, latents, t):
        assert t.dtype == torch.float32, f"t must be float32, but got {t.dtype}"
        assert latents.shape[0] == t.shape[0], "time must be given in batch manner"
        assert self.prompt_embeds is not None, "Prompt embeddings must be encoded first."

        return self.predict(latents, t)
    
    def get_tweedie(self, latents, vel_pred, t):
        assert t.dtype == torch.float32, f"t must be float32, but got {t.dtype}"
        latents = latents - vel_pred * t.reshape(t.shape[0], *(1,) * (latents.dim() - 1))
        return latents.to(self.pipe.dtype)

    def step(self, latents, t, next_t, vel_pred):
        assert t.dtype == torch.float32, f"t must be float32, but got {t.dtype}"
        assert next_t.dtype == torch.float32, f"next_t must be float32, but got {next_t.dtype}"

        dt = t - next_t
        dt = dt.reshape(dt.shape[0], *(1,) * (latents.dim() - 1))
        drift_coeff = -vel_pred

        next_latents = latents.to(torch.float32) + drift_coeff * dt
        return next_latents.to(self.pipe.dtype)
    
    def reverse_step(self, latents, t, next_t):
        s = next_t

        cur_scheduler_output = self.original_scheduler(t=t)
        next_scheduler_output = self.original_scheduler(t=s)

        alpha_t = cur_scheduler_output.alpha_t.to(latents.dtype)
        sigma_t = cur_scheduler_output.sigma_t.to(latents.dtype)
        alpha_s = next_scheduler_output.alpha_t.to(latents.dtype)
        sigma_s = next_scheduler_output.sigma_t.to(latents.dtype)
        
        alpha_t_s = alpha_t / alpha_s
        sigma_t_s = (sigma_t**2 - alpha_t_s**2 * sigma_s**2) ** 0.5
        
        latents = latents * alpha_t_s + sigma_t_s * torch.randn_like(latents)
        return latents.to(self.pipe.dtype)

    def decode_latents(self, latents, output_type="pil"):
        # If output_type is "pt", it returns a tensor with dtype torch.bfloat16

        assert self.height is not None and self.width is not None, "Check height and width are initialized"
        assert output_type in ["pt", "pil"], f"output_type must be 'pt' or 'pil', but got {output_type}"
        decoded_images = list()

        for i in range(0, latents.shape[0], self.cfg.batch_size):
            cur_batch_size = min(self.cfg.batch_size, latents.shape[0] - i)
            cur_latents = latents[i:i+cur_batch_size]

            cur_latents = self.pipe._unpack_latents(cur_latents, self.height, self.width, self.pipe.vae_scale_factor)
            cur_latents = (cur_latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
            cur_images = self.pipe.vae.decode(cur_latents, return_dict=False)[0]
            cur_images = self.pipe.image_processor.postprocess(cur_images, output_type=output_type)
            decoded_images.extend(cur_images) if output_type == "pil" else decoded_images.append(cur_images)

        if output_type == "pt":
            decoded_images = torch.cat(decoded_images, dim=0)

        return decoded_images

    def decode_latents_no_normalize(self, latents):
        decoded_pt = self.decode_latents(latents, output_type="pt")
        return 2.0 * (decoded_pt) - 1.0
    
    def get_reward_grad_vel_samples(self, latents, reward_model, t, return_grad=True):
        reward_list = list()
        grad_list = list()
        vel_pred_list = list()
        samples_list = list()

        for i in range(0, latents.shape[0], self.cfg.batch_size):
            cur_batch_size = min(self.cfg.batch_size, latents.shape[0] - i)
            cur_latents = latents[i : i + cur_batch_size].clone().detach().requires_grad_()
            cur_t = t[i : i + cur_batch_size]
            cur_vel_pred = self.forward(cur_latents, cur_t)
            samples = self.get_tweedie(cur_latents, cur_vel_pred, cur_t)
            if not reward_model.cfg.decode_to_unnormalized:
                decoded_samples = self.decode_latents(samples, output_type="pt")
            else:
                decoded_samples = self.decode_latents_no_normalize(samples)

            cur_reward_values = reward_model(decoded_samples.to(torch.float32), self)  # torch.float32

            if return_grad:
                cur_grad = torch.autograd.grad(cur_reward_values, cur_latents, torch.ones_like(cur_reward_values))[0]
                cur_grad = cur_grad.nan_to_num()
                
                if reward_model.cfg.grad_clip is not None:
                    assert reward_model.cfg.grad_norm is None, "grad_clip and grad_norm cannot be used together."
                    cur_grad = cur_grad.to(torch.float32)
                    cur_grad = torch.clamp(cur_grad, -reward_model.cfg.grad_clip, reward_model.cfg.grad_clip)
                    cur_grad = cur_grad.to(latents.dtype)
                elif reward_model.cfg.grad_norm is not None:
                    assert reward_model.cfg.grad_clip is None, "grad_clip and grad_norm cannot be used together."
                    cur_grad = cur_grad.to(torch.float32)
                    grad_scale = torch.mean(cur_grad ** 2) ** 0.5
                    if grad_scale > reward_model.cfg.grad_norm:
                        cur_grad = cur_grad * (reward_model.cfg.grad_norm / grad_scale)
                    cur_grad = cur_grad.to(latents.dtype)
                
                if hasattr(self.pipe.transformer, 'attn_maps'):
                    self.pipe.transformer.attn_maps = None

            reward_list.append(cur_reward_values.detach().clone())
            grad_list.append(cur_grad.detach().clone()) if return_grad else None
            vel_pred_list.append(cur_vel_pred.detach().clone())
            samples_list.append(decoded_samples.detach().clone())

            del cur_vel_pred, samples, decoded_samples, cur_reward_values


        rewards = torch.cat(reward_list, dim=0)
        grads = torch.cat(grad_list, dim=0) if return_grad else None
        vel_preds = torch.cat(vel_pred_list, dim=0)
        samples = torch.cat(samples_list, dim=0)
        if reward_model.cfg.decode_to_unnormalized:
            samples = samples * 0.5 + 0.5
        return rewards, grads, vel_preds, samples