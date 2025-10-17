from dataclasses import dataclass
from abc import ABC
import torch

from src.utils import ignore_kwargs


__STEP_SCHEDULER__ = dict()

def register_step_scheduler(name):
    def decorator(cls):
        __STEP_SCHEDULER__[name] = cls
        return cls
    return decorator

def get_step_scheduler(name):
    if name not in __STEP_SCHEDULER__:
        raise ValueError(f"Stepscheduler {name} not found. Available schedulers: {list(__STEP_SCHEDULER__.keys())}")
    return __STEP_SCHEDULER__[name]


class StepScheduler(ABC):
    @ignore_kwargs
    @dataclass
    class Config():
        pass

    def __init__(self, CFG):
        pass

@register_step_scheduler("uniform")
class UniformScheduler(StepScheduler):
    @ignore_kwargs
    @dataclass
    class Config(StepScheduler.Config):
        pass

    def __init__(self, CFG):
        self.cfg = self.Config(**CFG)

    def __call__(self, step_size, reward, grad_reward):
        # return step_scale, correction term 
        return 1.0, 0.0
    
@register_step_scheduler("adaptive")
class AdaptiveScheduler(StepScheduler):
    @ignore_kwargs
    @dataclass
    class Config(StepScheduler.Config):
        s_min: float = 1/3
        s_max: float = 4/3
        k: float = 0.3 

    def __init__(self, CFG):
        self.cfg = self.Config(**CFG)

    def __call__(self, step_size, reward, grad_reward):
        k, s_min, s_max = self.cfg.k, self.cfg.s_min, self.cfg.s_max
        
        sigmoid_output = 1. / (1. + torch.exp(2*k*reward))
        step_scale = s_min + (s_max - s_min) * (2*sigmoid_output - 1)
        correction_term = 2*k*(s_max - s_min) * sigmoid_output * (1. - sigmoid_output) * grad_reward * step_size
        return step_scale.to(grad_reward.dtype), correction_term.to(grad_reward.dtype)
        
        