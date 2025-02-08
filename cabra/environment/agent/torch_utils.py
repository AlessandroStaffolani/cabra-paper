import torch as th
from torch import nn as nn
from torch.nn.utils import skip_init


def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float):
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def init_layer(module_cls, *args, init_weights: bool = True, **kwargs) -> nn.Module:
    if init_weights:
        return module_cls(*args, **kwargs)
    else:
        return skip_init(module_cls, *args, **kwargs)
