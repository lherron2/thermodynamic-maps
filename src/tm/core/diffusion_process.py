import math
import torch
from torch import vmap
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def polynomial_noise(t, alpha_max, alpha_min, s=1e-5, **kwargs):
    """
    Generate polynomial noise schedule used in Hoogeboom et. al.

    Args:
        t (torch.Tensor): Time steps.
        alpha_max (float): Maximum alpha value.

        s (float): Smoothing factor.

    Returns:
        torch.Tensor: Alpha schedule.
    """
    T = t[-1]
    alphas = (1 - 2 * s) * (1 - (t / T) ** 2) + s
    a = alphas[1:] / alphas[:-1]
    a[a**2 < 0.001] = 0.001
    alpha_schedule = torch.cumprod(a, 0)
    betas = 1 - a
    return betas, alpha_schedule

def cosine_noise_schedule(t: torch.Tensor, s: float = 0.001, **kwargs):
    timesteps = kwargs.get("timesteps", t[-1])
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = alphas_cumprod[1:] / alphas_cumprod[:-1]    
    betas = 1 - torch.clamp(betas, 0, 0.999)
    alphas_cumprod = torch.clamp(alphas_cumprod, 0, 0.999)
    return betas, alphas_cumprod[:-1]

NOISE_FUNCS = {
    "polynomial": polynomial_noise,
    "cosine": cosine_noise_schedule,
}


class DiffusionProcess:
    """
    Instantiates the noise parameterization, rescaling of noise distribution,
    and timesteps for a diffusion process.
    """

    def __init__(
        self,
        num_diffusion_timesteps,
        noise_schedule,
        alpha_max,
        alpha_min,
        beta_start,
        beta_end,
        NOISE_FUNCS,
    ):
        """
        Initialize a DiffusionProcess.

        Args:
            num_diffusion_timesteps (int): Number of diffusion timesteps.
            noise_schedule (str): Noise schedule type.
            alpha_max (float): Maximum alpha value.
            alpha_min (float): Minimum alpha value.
            NOISE_FUNCS (dict): Dictionary of noise functions.
        """
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.times = torch.arange(num_diffusion_timesteps)
        self.betas, self.alphas = NOISE_FUNCS[noise_schedule](                
            t=torch.arange(num_diffusion_timesteps+1), alpha_max=alpha_max, alpha_min=alpha_min,
                beta_start=beta_start, beta_end=beta_end
            )

class VPDiffusion(DiffusionProcess):
    """
    Subclass of DiffusionProcess: Performs a diffusion according to the VP-SDE.
    """

    def __init__(
        self,
        num_diffusion_timesteps,
        noise_schedule="cosine",
        alpha_max=20.0,
        alpha_min=0.01,
        beta_start=0.001,
        beta_end=0.02,
        NOISE_FUNCS=NOISE_FUNCS,
    ):
        """
        Initialize a VPDiffusion process.

        Args:
            num_diffusion_timesteps (int): Number of diffusion timesteps.
            noise_schedule (str): Noise schedule type.
            alpha_max (float): Maximum alpha value.
            alpha_min (float): Minimum alpha value.
            NOISE_FUNCS (dict): Dictionary of noise functions.
        """
        super().__init__(
            num_diffusion_timesteps,
            noise_schedule,
            alpha_max,
            alpha_min,
            beta_start,
            beta_end,
            NOISE_FUNCS
        )
        self.bmul = vmap(torch.mul)

    def get_alphas(self):
        """
        Get alpha values.

        Returns:
            torch.Tensor: Alpha values.
        """
        return self.alphas

    def kernel(self, mode, *args, **kwargs):
        """
        Combined kernel method that handles forward, reverse, and jump kernels.
    
        Args:
            mode (str): Mode of operation ('forward', 'reverse', or 'jump').
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
    
        Returns:
            Depends on the mode:
                - 'forward': (x_t_next, noise, score)
                - 'jump': (x_{jump_t}, noise, score)
                - 'reverse': (x0_t, noise, score)
        """
        if mode == "jump":
            return self._forward_kernel_jump(*args, **kwargs)
        elif mode == "forward":
            if kwargs.get('likelihood', True):
                return self._forward_kernel_network_noise(*args, **kwargs)
            else:
                return self._forward_kernel_prior_noise(*args, **kwargs)
        elif mode == "reverse":
            return self._reverse_kernel(*args, **kwargs)
        elif mode == "reverse_jump":
            return self._reverse_kernel_jump(*args, **kwargs)
        else:
            raise ValueError("Mode must be 'forward', 'jump', or 'reverse'")

    def _forward_kernel_jump(self, x, t, prior, **prior_kwargs):
        """
        Modified forward marginal transition kernel that jumps from time t to jump_t in a single step.
    
        In our formulation, we first recover:
            x_0 = (x_t - sqrt(1 - α_t)*noise) / sqrt(α_t)
        and then jump directly to time jump_t:
            x_{jump_t} = sqrt(α_{jump_t}) * x_0 + sqrt(1 - α_{jump_t}) * noise
    
        Args:
            x_t (torch.Tensor): Data at time t.
            t (int): Current time step.
            jump_t (int): Target time step to jump to (must be greater than t).
            prior: Prior distribution used to sample noise.
            **prior_kwargs: Additional keyword arguments for the prior.
    
        Returns:
            tuple: (x_{jump_t}, noise, score)
        """
        alphas_t = self.alphas[t]
    
        noise = prior.sample(**prior_kwargs)
        x_jump = self.bmul(x, alphas_t.sqrt()) + self.bmul(noise, (1-alphas_t).sqrt())
        return x_jump, noise, None


    def _forward_kernel_prior_noise(self, x_t, t, prior, **prior_kwargs):
        """
        Modified forward marginal transition kernel that inverts the reverse diffusion step.
    
        Given:
          x0 = (x_t - sqrt(1-α_t)*noise) / sqrt(α_t)
        we form the next state as:
          x_{t+1} = sqrt(α_{t+1})*x0 + sqrt(1-α_{t+1})*noise
    
        Args:
            x_t (torch.Tensor): Data at time t.
            t (int): Current time step.
            prior: Prior distribution.
            **prior_kwargs: Additional keyword arguments.
    
        Returns:
            tuple: (x_{t+1}, noise, score)
        """
        beta_t = self.betas[t]
        noise = prior.sample(**prior_kwargs)
        x_t_next = self.bmul((1-beta_t).sqrt(), x_t) + self.bmul((beta_t).sqrt(), noise)    
        score = None
        
        return x_t_next, noise, score
    
    def _forward_kernel_network_noise(self, x_t, t, prior, backbone, network_pred_type, **kwargs):
        """
        Modified forward kernel using network-predicted noise or x0 to invert the reverse diffusion step.
    
        In the reverse kernel you recover:
            If pred_type=="noise":
               x0 = (x_t - sqrt(1-α_t)*noise) / sqrt(α_t)
            If pred_type=="x0":
               x0 = backbone(x_t, α_t)
        
        Then in the forward process we reconstruct:
             x_{t+1} = sqrt(α_{t+1}) * x0 + sqrt(1-α_{t+1}) * noise
    
        Args:
            x_t (torch.Tensor): Data at time t.
            t (int): Current time step.
            prior: Prior distribution (not used here but kept for API consistency).
            backbone: Backbone model.
            pred_type (str): Either 'noise' or 'x0'.
            **kwargs: Additional keyword arguments.
    
        Returns:
            tuple: (x_{t+1}, noise, score)
        """
        alpha_t = self.alphas[t]
        alpha_t_next = self.alphas[t + 1]
        beta_t = self.betas[t]
        
        if network_pred_type == "noise":
            noise = backbone(x_t, alpha_t)
            noise_rescaled = self.bmul(noise, (1 - alpha_t).sqrt())
            x0_t = self.bmul(x_t - noise_rescaled, 1 / (alpha_t).sqrt())

        elif network_pred_type == "x0":
            x0_t = backbone(x_t, alpha_t)
            noise = (x_t - x0_t * alpha_t.sqrt()) / (1 - alpha_t).sqrt()

        else:
            raise ValueError("network_pred_type must be 'noise' or 'x0'")

        # x_t_next = self.bmul((1-beta_t).sqrt(), x_t) + self.bmul((beta_t).sqrt(), noise)
        x_t_next = self.bmul(alpha_t_next.sqrt(), x0_t) + self.bmul((1 - alpha_t_next).sqrt(), noise)
        
        # Compute the score (adjust as necessary).
        score = None
        
        return x_t_next, noise, score

    def _reverse_kernel(self, x_t, t, backbone, network_pred_type, **kwargs):
        """
        Reverse marginal transition kernel p(x0 | x_t).

        Args:
            x_t (torch.Tensor): Data at time t.
            t (int): Time step.
            backbone: Backbone model.
            pred_type (str): Type of prediction ('noise' or 'x0').

        Returns:
            tuple: (x0_t, noise)
        """
        alpha_t = self.alphas[t]
        alpha_t_next = self.alphas[t-1]
        beta_t = self.betas[t]

        # reverse_variance = (1 - alpha_t_next)/(1 - alpha_t) * (1-beta_t)

        if network_pred_type == "noise":
            noise = backbone(x_t, alpha_t)
            noise_rescaled = self.bmul(noise, (1 - alpha_t).sqrt())
            x0_t = self.bmul(x_t - noise_rescaled,  1 / (alpha_t).sqrt())
            score = None
        elif network_pred_type == "x0":
            x0_t = backbone(x_t, alpha_t)
            noise = (x_t - x0_t * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
            score = None
        else:
            raise ValueError("Please provide a valid prediction type: 'noise' or 'x0'")
        
        x_t_next = self.bmul(alpha_t_next.sqrt(), x0_t) + self.bmul((1-alpha_t_next).sqrt(), noise)
        
        return x_t_next, noise, score

    def _reverse_kernel_jump(self, x_t, t, backbone, network_pred_type, **kwargs):
        """
        Reverse marginal transition kernel p(x0 | x_t).

        Args:
            x_t (torch.Tensor): Data at time t.
            t (int): Time step.
            backbone: Backbone model.
            pred_type (str): Type of prediction ('noise' or 'x0').

        Returns:
            tuple: (x0_t, noise)
        """
        alpha_t = self.alphas[t]
        alpha_t_next = self.alphas[t-1]
        beta_t = self.betas[t]

        reverse_variance = (1 - alpha_t_next)/(1 - alpha_t) * (1-beta_t)

        if network_pred_type == "noise":
            noise = backbone(x_t, alpha_t)
            noise_rescaled = self.bmul(noise, (1 - alpha_t).sqrt())
            x0_t = self.bmul((x_t - noise_rescaled), 1 / (alpha_t).sqrt())
            score = None
        elif network_pred_type == "x0":
            x0_t = backbone(x_t, alpha_t)
            noise = (x_t - x0_t * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
            score = None
        else:
            raise ValueError("Please provide a valid prediction type: 'noise' or 'x0'")
                
        return x0_t, noise, score

    def step(self, mode, x, t, t_next, backbone=None, network_pred_type=None, 
             prior=None, likelihood=False, control_dict=None, gamma=None, **kwargs):
        """
        Combined step method with forward and reverse steps that exactly invert one another
        (in the deterministic case).
    
        Args:
            mode (str): Operation mode: 'forward' or 'reverse'.
            x (torch.Tensor): Input data (x_t for forward; x_{t+1} for reverse).
            t (int): Current time step.
            t_next (int): Next time step for forward (t+1) or previous time step for reverse (t-1).
            backbone: Backbone model (used in reverse mode to predict noise or x0).
            pred_type (str): Either 'noise' or 'x0' (required for reverse mode).
            prior: Prior distribution (used in forward mode).
            likelihood (bool): If True, uses the network-based forward kernel.
            control_dict (dict): Optional channel-wise controls.
            gamma (float): Control strength.
            **kwargs: Additional keyword arguments.
    
        Returns:
            torch.Tensor: Updated data state (x_{t+1} for forward, x_{t-1} for reverse).
        """
        if mode == "forward":
            # The forward kernel (via self.kernel) returns (x_t_next, noise)
            x_next, noise, _ = self.kernel(
                mode="forward",
                x_t=x,
                t=t,
                prior=prior,
                likelihood=likelihood,
                backbone=backbone,
                network_pred_type=network_pred_type,
                **kwargs
            )
        elif mode == "reverse":
            # The reverse kernel returns (x0, noise)
            x_next, noise, _ = self.kernel(
                mode="reverse",
                x_t=x,
                t=t,
                backbone=backbone,
                network_pred_type=network_pred_type,
                **kwargs
            )

        else:
            raise ValueError("Mode must be 'forward' or 'reverse'")
    
        # Optionally apply channel-wise control.
        if control_dict:
            for channel, channel_control in control_dict.items():
                broadcast = torch.ones_like(x_next[:, channel])
                alpha_factor = gamma*broadcast*self.alphas[t_next][0]
                x_next[:, channel] = (1 - alpha_factor).sqrt() * x_next[:, channel] + alpha_factor.sqrt() * torch.tensor(channel_control)
    
        return x_next
