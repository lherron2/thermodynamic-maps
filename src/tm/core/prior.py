
import sys
import logging
import math
import torch
import numpy as np
from einops import rearrange
from scipy.stats import norm, expon
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class UnitNormalPrior:
    def __init__(self, shape, channels_info):
        self.channels_info = channels_info  # Dictionary to define channel types
        self.num_fluct_ch = len(self.channels_info['fluctuation'])
        self.num_coord_ch = len(self.channels_info['coordinate'])
        """Initialize the Unit Normal Prior with the shape of the samples."""
        self.shape = list(shape)[1:]
        logger.debug(f"Initialized a Prior with shape {self.shape}.")
        logger.debug(f"The first dimension of the supplied {shape=} must be the batch size.")

    def sample(self, batch_size, *args, **kwargs):
        """Sample from a unit normal distribution."""
        shape = [batch_size] + self.shape
        return torch.normal(mean=0, std=1, size=shape)

class GlobalEquilibriumHarmonicPrior(UnitNormalPrior):
    def __init__(self, shape, channels_info):
        """Initialize GEHP with shape and channels information."""
        super().__init__(shape, channels_info)

    def sample(self, batch_size, temperatures, *args, **kwargs):
        """Sample from a distribution where variance is defined by temperatures."""
        temperatures = torch.as_tensor(np.array(temperatures), dtype=torch.get_default_dtype())
        full_shape  = [batch_size] + self.shape
        coord_shape = [batch_size, self.num_coord_ch] + self.shape[1:]
        fluct_shape = [batch_size, self.num_fluct_ch] + self.shape[1:]
        # print(f"{full_shape=}")
        # print(f"{coord_shape=}")
        # print(f"{fluct_shape=}")
        samples = torch.empty(full_shape, dtype=temperatures.dtype)

        coord_variances = temperatures.expand(*coord_shape)

        fluct_variances = torch.ones(fluct_shape, dtype=coord_variances.dtype, device=coord_variances.device)
        variances = torch.cat((coord_variances, fluct_variances), dim=1)

        for sample_idx, ch_variances in enumerate(variances):
            samples[sample_idx] = torch.normal(mean=0., std=torch.sqrt(ch_variances))
        logger.debug(f"{samples.shape}")
        return samples

    def log_likelihood(self, samples, temperatures):
        """
        Log p(samples | temperatures) under zero-mean independent Gaussians.
        Coordinate channels get per-sample/channel variance from `temperatures`;
        fluctuation channels use unit variance.
        Returns: Tensor of shape [B] (sum over non-batch dims).
        """
        # Make tensors and collect shapes
        samples = torch.as_tensor(samples)
        dtype, device = samples.dtype, samples.device
        B = samples.shape[0]

        # Expected geometry from the class config
        # self.shape == [C_total, H, W, ...]; C_total = num_coord_ch + num_fluct_ch
        C_total = self.shape[0]
        assert C_total == (self.num_coord_ch + self.num_fluct_ch), \
            f"channels mismatch: total={C_total}, coord={self.num_coord_ch}, fluct={self.num_fluct_ch}"

        spatial_shape = list(self.shape[1:])  # e.g., [H, W, ...]
        coord_shape = [B, self.num_coord_ch] + spatial_shape
        fluct_shape = [B, self.num_fluct_ch] + spatial_shape

        # Prepare temperatures
        t = torch.as_tensor(temperatures, dtype=dtype, device=device)

        def to_coord_variances(tensor_t):
            """
            Convert various temperature shapes into a tensor that expands to coord_shape.
            Accepted minimal forms:
              - scalar
              - [B]
              - [B, 1]
              - [1, num_coord_ch]
              - [B, num_coord_ch]
            """
            if tensor_t.ndim == 0:
                # scalar -> [B,1,1,1,...]
                tensor_t = tensor_t.expand(B).view(B, *([1] * (len(coord_shape) - 1)))
            elif tensor_t.ndim == 1:
                if tensor_t.numel() == 1:  # [1] -> scalar case
                    tensor_t = tensor_t.expand(B).view(B, *([1] * (len(coord_shape) - 1)))
                elif tensor_t.numel() == B:  # [B] -> [B,1,1,1,...]
                    tensor_t = tensor_t.view(B, *([1] * (len(coord_shape) - 1)))
                else:
                    raise ValueError(f"temperatures shape [N]={list(tensor_t.shape)} incompatible with batch B={B}")
            elif tensor_t.ndim == 2:
                b, c = tensor_t.shape
                if b == B and c == self.num_coord_ch:
                    # [B, C_coord] -> [B, C_coord, 1, 1, ...]
                    tensor_t = tensor_t.view(B, self.num_coord_ch, *([1] * (len(coord_shape) - 2)))
                elif b == B and c == 1:
                    tensor_t = tensor_t.view(B, 1, *([1] * (len(coord_shape) - 2)))
                elif b == 1 and c == self.num_coord_ch:
                    tensor_t = tensor_t.expand(B, self.num_coord_ch).view(
                        B, self.num_coord_ch, *([1] * (len(coord_shape) - 2))
                    )
                elif b == 1 and c == 1:
                    tensor_t = tensor_t.expand(B, 1).view(B, 1, *([1] * (len(coord_shape) - 2)))
                else:
                    # Let broadcasting try, else fail below
                    pass

            # Final broadcast to coord_shape or error
            try:
                return tensor_t.expand(*coord_shape)
            except RuntimeError as e:
                raise ValueError(
                    f"temperatures with shape {list(t.shape)} cannot broadcast to {coord_shape}"
                ) from e

        coord_variances = to_coord_variances(t)
        fluct_variances = torch.ones(fluct_shape, dtype=dtype, device=device)
        variances = torch.cat((coord_variances, fluct_variances), dim=1)

        # Safety: positive variance
        variances = torch.clamp(variances, min=1e-12)

        if variances.shape != samples.shape:
            raise ValueError(
                f"Shape mismatch: variances {list(variances.shape)} vs samples {list(samples.shape)}"
            )

        # Elementwise log N(0, var): -0.5 * (x^2/var + log(2π var))
        log_probs = -0.5 * (samples.pow(2) / variances + torch.log(2 * math.pi * variances))
        # Sum over non-batch dims → [B]
        # return log_probs.flatten(start_dim=1).sum(dim=1)
        return log_probs.sum(dim=tuple(range(1, log_probs.ndim)))


class LocalEquilibriumHarmonicPrior(UnitNormalPrior):
    def __init__(self, shape, channels_info):
        """Initialize LEHP with shape and channels information."""
        super().__init__(shape, channels_info)

    @staticmethod
    def reshape(input):
        return rearrange(input, 'b h w c -> b c h w')

    def _reshape_for_fit_broadcast(self, temperatures, batch_size):
        """
        Replace the old 'multilinear fit' evaluate(): just make temperatures
        shape [B, 1, 1, 1, ...] (number of trailing singleton dims equals len(self.shape)).
        This allows broadcasting to [B, num_coord_ch, *self.shape[1:]].
        """
        temps = torch.as_tensor(temperatures, dtype=torch.get_default_dtype())
        temps = temps.view(-1)  # flatten
        if temps.numel() == 1:
            temps = temps.expand(batch_size)  # scalar -> per-sample
        elif temps.numel() != batch_size:
            raise ValueError(f"Expected 1 or {batch_size} temperatures but got {temps.numel()}")
        # [B, 1, 1, 1, ...] where number of trailing singletons = len(self.shape)
        temps = temps.view(batch_size, *([1] * len(self.shape)))
        return temps

    def sample(self, batch_size, temperatures, *args, **kwargs):
        """Sample from a distribution where variance is defined by temperatures."""
        # temperatures may be scalar, list/ndarray, or tensor
        temperatures = torch.as_tensor(temperatures, dtype=torch.get_default_dtype())

        full_shape  = [batch_size] + self.shape
        coord_shape = [batch_size, self.num_coord_ch] + self.shape[1:]
        fluct_shape = [batch_size, self.num_fluct_ch] + self.shape[1:]

        # just reshape temperatures so it broadcasts cleanly across coord_shape.
        if "sample_from_fit" in kwargs.values():
            temps_bc = self._reshape_for_fit_broadcast(temperatures, batch_size)
        else:
            # Generic broadcasting:
            # We convert to at least [B, 1, 1, 1, ...] to expand into coord_shape.
            t = temperatures
            if t.ndim == 0:
                t = t.expand(batch_size)                      # scalar -> [B]
            if t.ndim == 1:
                # [B] -> [B, 1, 1, 1, ...]
                t = t.view(batch_size, *([1] * len(self.shape)))
            temps_bc = t

        coord_variances = temps_bc.expand(*coord_shape)  # [B, C_coord, H, W]
        fluct_variances = torch.ones(fluct_shape, dtype=coord_variances.dtype, device=coord_variances.device)

        variances = torch.cat((coord_variances, fluct_variances), dim=1)
        std   = torch.sqrt(torch.clamp(variances, min=1e-12))
        mean  = torch.zeros_like(std)
        samples = torch.normal(mean=mean, std=std)
        return samples

    def log_likelihood(self, samples, temperatures, *args, **kwargs):
        # (fit branch removed; only broadcast reshape remains)
        samples = torch.as_tensor(samples, dtype=torch.get_default_dtype())
        B = samples.shape[0]
        temperatures = torch.as_tensor(temperatures, dtype=samples.dtype, device=samples.device)

        # make temperatures broadcast to coord shape
        if temperatures.ndim == 0:
            temperatures = temperatures.expand(B)
        if temperatures.ndim == 1:
            temperatures = temperatures.view(B, *([1] * (samples.ndim - 1)))  # [B,1,1,1,...]

        coord_shape = [B, self.num_coord_ch] + list(self.shape[1:])
        coord_var = temperatures.expand(coord_shape)

        fluct_shape = [B, self.num_fluct_ch] + list(self.shape[1:])
        fluct_var = torch.ones(fluct_shape, dtype=samples.dtype, device=samples.device)

        variances = torch.cat((coord_var, fluct_var), dim=1)
        variances = torch.clamp(variances, min=1e-12)

        if variances.shape != samples.shape:
            raise ValueError(f"Shape mismatch: variances {variances.shape} vs samples {samples.shape}")

        log_probs = -0.5 * ((samples ** 2) / variances + torch.log(2 * math.pi * variances))
        return log_probs.sum(dim=tuple(range(1, log_probs.ndim)))




