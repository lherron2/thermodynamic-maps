import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce

import logging

# helper functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
        
class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    Weight standardization works well with group normalization.
    """
    def forward(self, x):
        # Set epsilon based on dtype.
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3


        weight = self.weight

        # Compute mean over non-output channel dimensions using einops.
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        sample_mean = mean.flatten()[0].item() if mean.numel() > 0 else 'N/A'

        # Instead of using a custom reduction function to compute the variance,
        # we compute the variance for each output channel manually over the remaining dimensions.
        # For a weight tensor of shape [o, c, k], we want the variance over the c and k dimensions.
        var_manual = torch.var(weight, dim=list(range(1, weight.ndim)), unbiased=False).view(weight.shape[0], 1, 1)
        sample_var = var_manual.flatten()[0].item() if var_manual.numel() > 0 else 'N/A'

        # Normalize the weights.
        normalized_weight = (weight - mean) * (var_manual + eps).rsqrt()

        # Apply the convolution and log the output shape.
        conv_output = F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        return conv_output


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeddings

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """Option for either random or learned sinusoidal positional embeddings."""
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)
    
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h, scale_shift=scale_shift)
        return h + self.res_conv(x)


# The updated Unet1D class without attention and up/down sampling.
class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        num_resolutions=4,  # number of levels
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        **kwargs
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        # All blocks will now keep the same number of channels.
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        # time embedding
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Build down-sampling blocks (without downsampling or attention)
        self.downs = nn.ModuleList([])
        for _ in range(num_resolutions):
            self.downs.append(nn.ModuleList([
                block_klass(init_dim, init_dim, time_emb_dim=time_dim),
                block_klass(init_dim, init_dim, time_emb_dim=time_dim),
            ]))

        # Middle blocks, with mid attention removed.
        self.mid_block1 = block_klass(init_dim, init_dim, time_emb_dim=time_dim)
        self.mid_attn = nn.Identity()  # attention removed
        self.mid_block2 = block_klass(init_dim, init_dim, time_emb_dim=time_dim)

        # Build up-sampling blocks (without upsampling or attention)
        self.ups = nn.ModuleList([])
        for _ in range(num_resolutions):
            # Each up level gets a concatenated input of (x, skip) so input channels = init_dim * 2.
            self.ups.append(nn.ModuleList([
                block_klass(init_dim * 2, init_dim, time_emb_dim=time_dim),
                block_klass(init_dim * 2, init_dim, time_emb_dim=time_dim),
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # Final layers combine the result with the initial conv features.
        self.final_res_block = block_klass(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        h = []

        # Down path: apply two Resnet blocks per level and store outputs for skip connections.
        for block1, block2 in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            h.append(x)
            # Downsampling removed: x remains unchanged

        # Middle part.
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)  # Identity here.
        x = self.mid_block2(x, t)

        # Up path: for each level, concatenate skip connections and apply blocks.
        for block1, block2 in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            # Upsampling removed: x remains unchanged

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
