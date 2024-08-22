"""
Modules for Unet Model.
"""

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

import einx
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange

from functools import partial

from utils import default, exists

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0

def Upsample(dim, dim_out = None):
    # Flexible upsample for Unet.ups modules
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    # Custom downsample for Unet.downs
    return nn.Sequential(
        #  3, (2 128), (2, 128) -> (12) 64 64'
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    # https://arxiv.org/abs/1910.07467
    # https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    # regularizes summed inputs according to the root mean square statistic
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        # Equation 4 in the paper, a_i / RMS(a) * gamma * scale factor
        # first part is available in nn.functional.normalize, with euclidean norm as default
        return F.normalize(x, dim = 1) * (self.gamma + 1) * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    # From Attention is All you need, equation 3.5 https://arxiv.org/pdf/1706.03762
    # Equips each token with information about its position in a sequence, dim-dimensional vector.
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = einx.multiply('i, j -> i j', x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    # Also proposed in https://arxiv.org/pdf/1706.03762
    # following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb
    # https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)
        # DAFT https://arxiv.org/abs/2107.05990 uses this but this was introduced in 
        # FiLM https://arxiv.org/abs/1709.07871 as Feature wise linear modulation.
        # Equation 2 in FiLM: Feature_out = gamma * Feature_in + beta
        # used in ResnetBlock to allow time information to shift & scale features
        # dynamically and propagate positional info through the network.
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    # Resnet block with time MLP & res_conv layer 
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    """
    Attention with Linear Complexity by sequence length
    https://github.com/lucidrains/linear-attention-transformer
    https://github.com/lucidrains/linear-attention-transformer/blob/24ecf20b11a7c8ddbc15e33a30f0be0cc73b145d/linear_attention_transformer/linear_attention_transformer.py#L204
    https://arxiv.org/abs/2006.16236
    https://arxiv.org/abs/1812.01243
    https://arxiv.org/abs/2006.04768
    """
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = tuple(rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads) for t in qkv)

        mk, mv = tuple(repeat(t, 'h c n -> b h c n', b = b) for t in self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = einsum(k, v, 'b h d n, b h e n -> b h d e')

        out = einsum(context, q, 'b h d e, b h d n -> b h e n')
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        # additional normalization
        return self.to_out(out)

class Attention(Module):
    """
    Standard Attention 
    """
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, bias = False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

        
