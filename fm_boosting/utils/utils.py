import os
import shutil

import torch
from torch import pi

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

import torchvision
from torchvision.models import VGG16_Weights

from einops import reduce

from torchdiffeq import (
    odeint,
    odeint_adjoint,
)


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    """
    Print memory requirement estimate from model parameters.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


class WrapperCondFlow(nn.Module):
    # Hacky wrapper to add conditioning to UNet, should probably should write proper wrapper like: 
    # class SuperResModel(UNetModel): etc
    def __init__(self, model, cond) -> None:
        super().__init__()
        self.model = model
        self.cond = cond
    def forward(self, t, x):
        x = torch.cat([x, self.cond], 1)
        return self.model(t, x)


def broadcast_params(params):
    """
    For distributed training.
    """
    for param in params:
        dist.broadcast(param.data, src=0)


def sample_from_model(model, z_0):
    """
    Sample from a model during training.
    Use ode solver to integrate along path and return the image obtained at end of path.
    TODO: Integrate this into a trainer class like in https://github.com/lucidrains/rectified-flow-pytorch/blob/598d08eb687d6f5eee292874bf2ad60d67e2a293/rectified_flow_pytorch/rectified_flow.py#L965
    """
    t = torch.tensor([1.0, 0.0], device="cuda")
    fake_image = odeint(model, z_0, t, atol=1e-8, rtol=1e-8)
    return fake_image


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

# losses

class LPIPSLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])

        self.vgg = [vgg]

    def forward(self, pred_data, data, reduction = 'mean'):
        vgg, = self.vgg
        vgg = vgg.to(data.device)

        pred_embed, embed = map(vgg, (pred_data, data))

        loss = F.mse_loss(embed, pred_embed, reduction = reduction)

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

class PseudoHuberLoss(Module):
    def __init__(self, data_dim: int = 3):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, pred, target, reduction = 'mean', **kwargs):
        data_dim = default(self.data_dim, kwargs.pop('data_dim', None))

        c = .00054 * self.data_dim
        loss = (F.mse_loss(pred, target, reduction = reduction) + c * c).sqrt() - c

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

class PseudoHuberLossWithLPIPS(Module):
    def __init__(self, data_dim: int = 3, lpips_kwargs: dict = dict()):
        super().__init__()
        self.pseudo_huber = PseudoHuberLoss(data_dim)
        self.lpips = LPIPSLoss(**lpips_kwargs)

    def forward(self, pred_flow, target_flow, *, pred_data, times, data):
        huber_loss = self.pseudo_huber(pred_flow, target_flow, reduction = 'none')
        lpips_loss = self.lpips(data, pred_data, reduction = 'none')

        time_weighted_loss = huber_loss * (1 - times) + lpips_loss * (1. / times.clamp(min = 1e-1))
        return time_weighted_loss.mean()

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)
