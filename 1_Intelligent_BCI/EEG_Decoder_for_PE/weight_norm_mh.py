import copy
import torch
import numpy as np
from torch import nn
import random
from dn3_utils_mh import min_max_normalize

import mne
import parse
import tqdm
import torch.nn.functional as F
from math import ceil
from pathlib import Path



# for control the random seed
random_seed = 2022
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

from torch.nn.parameter import Parameter#, UninitializedParameter
from torch import norm_except_dim
from typing import Any, TypeVar
import torch.nn as nn

# __all__ = ['WeightNorm', 'weight_norm', 'remove_weight_norm']

class WeightNorm_mh(nn.Module):
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: nn.Module, dim) -> Any:
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        w = v * (g / (torch.norm_except_dim(v, 2, dim) + 1e-8)).expand_as(v)
        return w #_weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int) -> 'WeightNorm_mh':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm_mh) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = WeightNorm_mh(name, dim)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(norm_except_dim(weight, 2, dim).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module, dim))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: nn.Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module, dim=2))


T_module = TypeVar('T_module', bound=nn.Module)
#
def weight_norm_mh(module: T_module, name: str = 'weight', dim: int = 0) -> T_module:
    """Applies weight normalization to a parameter in the given module.
    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}
    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.
    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.
    See https://arxiv.org/abs/1602.07868
    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm
    Returns:
        The original module with the weight norm hook
    """
    WeightNorm_mh.apply(module, name, dim)
    return module

#
# def remove_weight_norm(module: T_module, name: str = 'weight') -> T_module:
#     r"""Removes the weight normalization reparameterization from a module.
#     Args:
#         module (Module): containing module
#         name (str, optional): name of weight parameter
#     Example:
#         >>> m = weight_norm(nn.Linear(20, 40))
#         >>> remove_weight_norm(m)
#     """
#     for k, hook in module._forward_pre_hooks.items():
#         if isinstance(hook, WeightNorm) and hook.name == name:
#             hook.remove(module)
#             del module._forward_pre_hooks[k]
#             return module
#
#     raise ValueError("weight_norm of '{}' not found in {}"
#                      .format(name, module))