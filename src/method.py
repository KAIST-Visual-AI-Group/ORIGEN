from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import partial
from tqdm import tqdm

import torch

from src.utils import ignore_kwargs
from src.step_scheduler import get_step_scheduler


__SAMPLER__ = dict()

def register_sampler(name):
    def decorator(cls):
        __SAMPLER__[name] = cls
        return cls
    return decorator

def get_sampler(name):
    if name not in __SAMPLER__:
        raise ValueError(f"Sampler {name} not found. Available samplers: {list(__SAMPLER__.keys())}")
    return __SAMPLER__[name]


__CALL_FUNCTIONS__ = dict()

def register_call_function(name):
    def decorator(func):
        __CALL_FUNCTIONS__[name] = func
        return func
    return decorator

def get_call_function(name):
    if name not in __CALL_FUNCTIONS__:
        raise ValueError(f"Call function {name} not found. Available functions: {list(__CALL_FUNCTIONS__.keys())}")
    return __CALL_FUNCTIONS__[name]


@register_call_function("all")
def get_all_samples(samples, rewards, **kwargs):
    true_max_idx = torch.argmax(rewards, dim=0)
    best_sample = samples[true_max_idx:true_max_idx + 1]
    true_max_reward = rewards[true_max_idx]
    return samples, best_sample, true_max_reward


class BaseSampler(ABC):
    @ignore_kwargs
    @dataclass
    class Config():
        step_size: float = 0.3
        eta: float = 0.8
        num_steps: int = 50

        custom_call_function_name: str = "all"
        step_scheduler: str = "adaptive"     #  uniform, adaptive
        early_stop: bool = True

    @abstractmethod
    def __init__(self, CFG):
        self.cfg = self.Config(**CFG)
        if self.cfg.custom_call_function_name is not None:
            self.custom_call_function = get_call_function(self.cfg.custom_call_function_name)
            
            cfg_dict = {"num_steps": self.cfg.num_steps}
            
            self.custom_call_function = partial(self.custom_call_function, **cfg_dict)
        else:
            self.custom_call_function = lambda x, y: x
        self.step_scheduler = get_step_scheduler(self.cfg.step_scheduler)(CFG)
        
    @abstractmethod
    def run(self, init_position, grad_reward):
        pass

    def get_stepsize_and_correction_term(self, reward, grad_reward):
        step_scale, correction_term = self.step_scheduler(self.cfg.step_size, reward, grad_reward)
        return self.cfg.step_size * step_scale, correction_term

    def get_grad_reward_func(self, reward_model, pipe, **kwargs):
        def grad_reward_func(latents, pipe, **kwargs):
            """
            NOTE: It returns reward and gradient of reward not the ones divided with alpha_mcmc.
            It is your responsibility to divide them with alpha_mcmc if you want to use them in the MCMC.
            """
            init_t = torch.tensor([1.0], device=latents.device, dtype=torch.float32).expand(latents.shape[0])
            reward_val, grad_reward, _, decoded_samples = pipe.get_reward_grad_vel_samples(latents, reward_model, init_t)
            return grad_reward, reward_val, decoded_samples

        return partial(grad_reward_func, pipe=pipe, **kwargs)
    
    def __call__(self, init_position, reward_model, pipe):
        samples, rewards = self.run(init_position, self.get_grad_reward_func(reward_model, pipe), reward_model)
        samples, best_sample, best_reward = self.custom_call_function(samples, rewards)
        best_sample = reward_model.success_sample if self.cfg.early_stop and reward_model.success else best_sample
        self.best_sample = best_sample
        self.best_reward = best_reward
        self.rewards = rewards
        return samples, best_sample, best_reward
    
    def set_custom_call_function(self, func):
        self.custom_call_function = func

    def get_tqdm(self):
        return tqdm(range(self.cfg.num_steps), desc="Initial Sampling", leave=False, total=self.cfg.num_steps)


@register_sampler("origen")
class ORIGEN(BaseSampler):
    @ignore_kwargs
    @dataclass
    class Config(BaseSampler.Config):
        pass

    def __init__(self, CFG):
        self.cfg = self.Config(**CFG)
        self.first_iter = None
        super().__init__(CFG)
    
    def run(self, init_position, grad_reward):
        original_dim = init_position.shape
        init_position = init_position.reshape(original_dim[0], -1)
        samples = list()
        reward_list = list()
        x_current = init_position.clone()

        tqdm_obj = self.get_tqdm()
        for i in tqdm_obj:
            cur_grad, reward, sample = grad_reward(x_current.reshape(original_dim))
                    
            samples.append(sample.clone().detach())
            reward_list.append(reward.clone().detach())

            step_size, correction_term = self.get_stepsize_and_correction_term(reward, cur_grad)
            cur_grad = cur_grad.reshape(original_dim[0], -1)
            correction_term = correction_term.reshape(original_dim[0], -1) if isinstance(correction_term, torch.Tensor) else correction_term
            
            x_current = (1. - step_size)**0.5*x_current + self.cfg.eta * step_size * cur_grad + (step_size**0.5) * torch.randn_like(x_current) + correction_term

        _, reward, sample = grad_reward(x_current.reshape(original_dim))
                
        reward_list.append(reward.clone().detach())
        samples.append(sample.clone().detach())
        return torch.cat(samples, dim=0).reshape(len(samples), *sample.shape[1:]), torch.cat(reward_list, dim=0).reshape(len(samples))