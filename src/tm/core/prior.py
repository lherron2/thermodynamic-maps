
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
        temperatures = torch.Tensor(np.array(temperatures))
        full_shape = [batch_size] + self.shape
        coord_shape = [batch_size] + [self.num_coord_ch] + self.shape[1:]
        fluct_shape = [batch_size] + [self.num_fluct_ch] + self.shape[1:]
        samples = torch.empty(full_shape)

        temps_for_each_channel_bool = temperatures.shape[1] == self.num_coord_ch 
        single_temp_provided_bool = temperatures.shape[0] == 1
        temps_for_each_sample_in_batch_bool = temperatures.shape[0] == batch_size

        if not temps_for_each_channel_bool and temps_for_each_sample_in_batch_bool:
            temperatures = temperatures.unsqueeze(-1).unsqueeze(-1).expand(*coord_shape)
            coord_variances = temperatures # expand along batch and coordinate dims
        else:
            coord_variances = temperatures.unsqueeze(-1).unsqueeze(-1).expand(*coord_shape) # expand along batch and coordinate dims
        fluct_variances = torch.full((fluct_shape), 1)

        variances = torch.cat((coord_variances, fluct_variances), dim=1)

        for sample_idx, ch_variances in enumerate(variances):
            # logging.debug(f"{ch_variances.shape}")
            samples[sample_idx] = torch.normal(mean=0., std=np.sqrt(ch_variances))
        logger.debug(f"{samples.shape}")
        return samples

class LocalEquilibriumHarmonicPrior(UnitNormalPrior):
    def __init__(self, shape, channels_info, multilinear_fit):
        """Initialize LEHP with shape, channels information, and MultiLinearFit."""
        super().__init__(shape, channels_info)
        self.multilinear_fit = multilinear_fit

    @staticmethod
    def reshape(input):
        return rearrange(input, 'b h w c -> b c h w')

    def sample(self, batch_size, temperatures, *args, **kwargs):
        """Sample from a distribution where variance is defined by temperatures."""
        if hasattr(temperatures, "__len__"): # If it is an array
            temperatures = torch.Tensor(temperatures)
        else: # Otherwise it's a scalar
            temperatures = torch.Tensor(np.array([temperatures]))
         
        full_shape = [batch_size] + self.shape
        coord_shape = [batch_size] + [self.num_coord_ch] + self.shape[1:]
        fluct_shape = [batch_size] + [self.num_fluct_ch] + self.shape[1:]

        samples = torch.empty(full_shape)

        if "sample_from_fit" in kwargs.values():
            temperatures = self.multilinear_fit.evaluate(temperatures)
            temperatures = torch.Tensor(temperatures)
            
        coord_variances = temperatures.expand(*coord_shape) # expand along batch and coordinate dims
        fluct_variances = torch.full((fluct_shape), 1)

        variances = torch.cat((coord_variances, fluct_variances), dim=1)
        std   = torch.sqrt(variances)                # [B, D]
        mean  = torch.zeros_like(std)                # [B, D]
        samples = torch.normal(mean=mean, std=std)
            
        return samples

    def log_likelihood(self, samples, temperatures, *args, **kwargs):
        # 1) ensure samples is a tensor (keep its original dtype/device)
        samples = torch.as_tensor(samples, dtype=torch.get_default_dtype(), device=None)
        B = samples.shape[0]
    
        # 2) get temperatures as a tensor on the same device/dtype as samples
        temperatures = torch.as_tensor(temperatures, dtype=samples.dtype, device=samples.device)
    
        # 3) optional multilinear fit
        if "sample_from_fit" in kwargs.values():
            # evaluate often returns a numpy array; rewrap as tensor
            fit_vals = self.multilinear_fit.evaluate(temperatures.cpu().numpy())
            temperatures = torch.as_tensor(fit_vals, dtype=samples.dtype, device=samples.device)
    
        # 4) flatten to 1-D
        temperatures = temperatures.view(-1)
    
        # 5) broadcast or validate
        if temperatures.numel() == 1:
            # one scalar → expand to [B]
            temperatures = temperatures.expand(B)
        elif temperatures.numel() != B:
            raise ValueError(f"Expected 1 or {B} temperatures but got {temperatures.numel()}")
    
        # 6) build per-coordinate variances
        coord_shape = [B, self.num_coord_ch] + list(self.shape[1:])
        coord_var = temperatures.view(B, 1, *([1] * (samples.ndim - 2))).expand(coord_shape)
    
        fluct_shape = [B, self.num_fluct_ch] + list(self.shape[1:])
        fluct_var = torch.ones(fluct_shape, dtype=samples.dtype, device=samples.device)
    
        variances = torch.cat((coord_var, fluct_var), dim=1)
        variances = torch.clamp(variances, min=1e-12)
    
        if variances.shape != samples.shape:
            raise ValueError(f"Shape mismatch: variances {variances.shape} vs samples {samples.shape}")
    
        # 7) compute log‐prob per element
        log_probs = -0.5 * (
            (samples ** 2) / variances
            + torch.log(2 * math.pi * variances)
        )
    
        # 8) sum over all but the batch dimension
        return log_probs.sum(dim=tuple(range(1, log_probs.ndim)))



    # def log_likelihood(self, samples, temperatures, *args, **kwargs):
    #     """Compute the log-likelihood of the samples given the temperatures."""
    #     if hasattr(temperatures, "__len__"):  # If it is an array
    #         temperatures = torch.Tensor(temperatures)
    #     else:  # Otherwise it's a scalar
    #         temperatures = torch.Tensor([temperatures])

    #     batch_size = samples.shape[0]
    #     coord_shape = [batch_size, self.num_coord_ch] + self.shape[1:]
    #     fluct_shape = [batch_size, self.num_fluct_ch] + self.shape[1:]

    #     if "sample_from_fit" in kwargs.values():
    #         temperatures = self.multilinear_fit.evaluate(temperatures)
    #         temperatures = torch.Tensor(temperatures)

    #     coord_variances = temperatures.expand(*coord_shape)
    #     fluct_variances = torch.ones(fluct_shape)

    #     variances = torch.cat((coord_variances, fluct_variances), dim=1)

    #     # Ensure variances and samples have the same shape
    #     if variances.shape != samples.shape:
    #         raise ValueError("Variances and samples must have the same shape.")

    #     # Compute the log-likelihood
    #     log_probs = -0.5 * ((samples ** 2) / variances + torch.log(2 * np.pi * variances))

    #     return log_probs


# class MultiLinearFit:
#     def __init__(self, min_value):
#         self.exp_loc = None    # Exponential distribution location parameter for each pixel/channel
#         self.exp_scale = None  # Exponential distribution scale parameter for each pixel/channel
#         self.min_value = min_value
#         self.temp_mu = None    # Mean of the fitted Gaussian for temperatures
#         self.temp_sigma = None # Standard deviation of the fitted Gaussian for temperatures

#     def fit(self, rmsfs, temperatures, dims=0):
#         """
#         Fit the probability integral transform model. This involves:
#           - Fitting a Gaussian to the temperature data.
#           - Fitting an exponential distribution to the rmsf values for each pixel (and channel).
          
#         Parameters:
#         -----------
#         rmsfs : numpy array
#             The rmsf data, with shape depending on 'dims':
#               - dims == 4: (n_samples, N, N, n_channels)
#               - dims == 3: (n_samples, N, n_channels)
#               - dims == 2: (n_samples, N)
#         temperatures : array-like
#             Temperature values for each sample (n_samples,).
#         dims : int, optional
#             Dimensionality of the rmsf data. If 0, defaults to 4.
#         """
#         # Add a slight random perturbation to avoid issues with identical temperature values
#         temperatures = np.array(temperatures) + 1e-3 * np.random.uniform(low=-0.5, high=0.5, size=len(temperatures))
#         rmsfs = np.array(rmsfs)
        
#         # Fit a Gaussian to the temperature data (kept for compatibility, though not used in evaluate)
#         self.temp_mu = temperatures.mean()
#         self.temp_sigma = temperatures.std()
        
#         if dims == 0:
#             dims = 4  # default assumption
        
#         if dims == 4:
#             n_samples, N, _, n_c = rmsfs.shape
#         elif dims == 3:
#             n_samples, N, n_c = rmsfs.shape
#         elif dims == 2:
#             n_samples, N = rmsfs.shape
#             n_c = 1
#         else:
#             raise ValueError("dims must be at least 2 (e.g. [n_samples, data_dim]).")
        
#         # Initialize storage for the exponential parameters
#         if dims == 4:
#             self.exp_loc = np.zeros((N, N, n_c))
#             self.exp_scale = np.zeros((N, N, n_c))
#             for i in range(N):
#                 for j in range(N):
#                     for c in range(n_c):
#                         pixel_values = rmsfs[:, i, j, c]
#                         # Fit an exponential distribution to the pixel's rmsf values.
#                         loc, scale = expon.fit(pixel_values)
#                         self.exp_loc[i, j, c] = loc
#                         self.exp_scale[i, j, c] = scale
#         elif dims <= 3:
#             self.exp_loc = np.zeros((N, n_c))
#             self.exp_scale = np.zeros((N, n_c))
#             for i in range(N):
#                 for c in range(n_c):
#                     if dims == 3:
#                         pixel_values = rmsfs[:, i, c]
#                     else:  # dims == 2
#                         pixel_values = rmsfs[:, i]
#                     loc, scale = expon.fit(pixel_values)
#                     self.exp_loc[i, c] = loc
#                     self.exp_scale[i, c] = scale

#     def evaluate(self, temperatures):
#         """
#         Evaluate the model for given temperature(s). This modified version
#         ignores the previously fitted exponential and Gaussian parameters and 
#         simply returns the input temperature, reshaped as a constant array matching 
#         the shape of a single fitted image (i.e. same shape as self.exp_loc).

#         Parameters:
#         -----------
#         temperatures : scalar or array-like
#             Temperature value(s) to output.

#         Returns:
#         --------
#         predicted_images : numpy array
#             The output images, where each image is filled with the corresponding 
#             temperature value. If a scalar is given, the output will have a new 
#             sample axis added.
#         """
#         temperatures = np.array(temperatures)

#         def make_image(t):
#             out = t/self.exp_scale
#             return out

#         if temperatures.ndim == 0:
#             # If a scalar, return a single sample image
#             return np.expand_dims(make_image(temperatures), axis=0)
#         else:
#             # For each temperature, create an image of the same shape as self.exp_loc
#             images = [make_image(t) for t in temperatures]
#             return np.stack(images, axis=0)


import torch

class MultiLinearFit:
    def __init__(self, min_value=None):
        self.exp_loc: torch.Tensor    # Exponential location parameter, shape depends on dims
        self.exp_scale: torch.Tensor  # Exponential scale parameter
        self.min_value = min_value
        self.temp_mu: torch.Tensor    # Mean of the fitted temperatures
        self.temp_sigma: torch.Tensor # Stddev of the fitted temperatures

    def fit(self, rmsfs, temperatures, dims=0):
        """
        Tensor-based fit:
          - Gaussian fit to temperatures (stored but not used in evaluate)
          - Exponential fit to RMSF values per pixel/channel via MLE:
            loc = min(rmsf), scale = mean(rmsf - loc)
        """
        # to tensor
        temps = torch.as_tensor(temperatures, dtype=torch.float32)
        # tiny jitter to break ties
        temps = temps + 1e-3 * (torch.rand_like(temps) - 0.5)

        data = torch.as_tensor(rmsfs, dtype=torch.float32)
        # Gaussian fit
        self.temp_mu = temps.mean()
        self.temp_sigma = temps.std()

        # infer dims
        if dims == 0:
            dims = 4
        if dims == 4:
            n_samples, N, _, n_c = data.shape
        elif dims == 3:
            n_samples, N, n_c   = data.shape
        elif dims == 2:
            n_samples, N        = data.shape
            n_c = 1
            data = data.unsqueeze(-1)  # make shape (n_samples, N, 1)
        else:
            raise ValueError("dims must be 2, 3, or 4.")

        # now data has shape (n_samples, *spatial_dims, n_c)
        spatial_shape = data.shape[1:-1]  # e.g. (N,N) or (N,)
        # compute loc = min over samples, scale = mean(data - loc)
        # data.min(dim=0) returns (values, indices)
        loc = data.min(dim=0)[0]  # shape = (*spatial_dims, n_c)
        scale = (data - loc).mean(dim=0)  # same shape

        self.exp_loc = loc
        self.exp_scale = scale

    def evaluate(self, temperatures):
        """
        For each temperature t, returns a constant image filled with t/self.exp_scale.
        Output is a tensor of shape (n_temps, *spatial_dims, n_c).
        """
        temps = torch.as_tensor(temperatures, dtype=self.exp_scale.dtype)
        # ensure a batch dimension
        if temps.dim() == 0:
            temps = temps.unsqueeze(0)  # shape (1,)

        # reshape for broadcasting: (n, 1, ..., 1)
        # number of trailing singleton dims == exp_scale.ndim
        shape = temps.shape + (1,) * self.exp_scale.ndim
        temps_reshaped = temps.view(shape)

        return temps_reshaped




