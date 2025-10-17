from dataclasses import dataclass, field

import torch
from torch import Tensor


__SCHEDULER__ = dict()

def register_scheduler(name):
    def decorator(cls):
        __SCHEDULER__[name] = cls
        return cls
    return decorator

def get_scheduler(name):
    if name not in __SCHEDULER__:
        raise ValueError(f"Scheduler {name} not found. Available schedulers: {list(__SCHEDULER__.keys())}")
    return __SCHEDULER__[name]


@dataclass
class SchedulerOutput:
    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})

@register_scheduler("linear")
class LinearScheduler():
    def __init__(self):
        pass

    def __call__(self, t):
        return SchedulerOutput(alpha_t = 1 - t, 
                               sigma_t = t, 
                               d_alpha_t = -torch.ones_like(t), 
                               d_sigma_t = torch.ones_like(t))

    def snr_inverse(self, snr):
        return 1.0 / (1.0 + snr)
        