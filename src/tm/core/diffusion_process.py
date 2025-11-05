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


# def cosine_noise_schedule(t: torch.Tensor, s: float = 0.001, **kwargs):
#     timesteps = kwargs.get("timesteps", t[-1])

#     # Raw cumulative alphas
#     alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     alphas_cumprod = torch.clamp(alphas_cumprod, 0.0, 0.999)
    
#     alphas_ratio = alphas_cumprod[1:] / alphas_cumprod[:-1]
#     betas = 1.0 - alphas_ratio             # derive betas from clamped alphas
#     betas = torch.clamp(betas, 0.0, 0.999)
    
#     return betas, alphas_cumprod[:-1]

def cosine_noise_schedule(t: torch.Tensor, s: float = 0.001, **kwargs):
    """
    Returns:
        betas[t]          = β_t, shape (T,)
        alphas_cumprod[t] = \bar{α}_t, shape (T,)
    where T = len(t) - 1.
    """
    T = t[-1].item()  # e.g. T
    steps = torch.arange(T + 1, device=t.device, dtype=t.dtype)  # 0..T

    raw = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    raw = raw / raw[0]
    raw = torch.clamp(raw, 0.0, 0.999)

    # bar_alpha_0..bar_alpha_{T-1}: length T
    alphas_cumprod = raw[:-1]                           # (T,)

    # per-step ratio a_t = bar_alpha_{t+1} / bar_alpha_t
    a = raw[1:] / raw[:-1]                              # (T,)
    betas = torch.clamp(1.0 - a, 0.0, 0.999)            # (T,)

    return betas, alphas_cumprod


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
        self.times = torch.arange(num_diffusion_timesteps)  # 0..T-1

        betas, alphas_cumprod = NOISE_FUNCS[noise_schedule](
            t=torch.arange(num_diffusion_timesteps + 1),
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # shapes: (T,)
        self.betas = betas
        self.alphas = 1.0 - self.betas          # per-step α_t
        self.alphas_cumprod = alphas_cumprod    # bar_alpha_t, length T


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
        # self.bmul = vmap(torch.mul)
        
    @staticmethod
    def _scale_like(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Broadcast scalar/timestep coefficients s to have same shape as x
        along non-batch dimensions and multiply.
        s is assumed to have shape (B,) or ().
        """
        if s.ndim == 0:
            return x * s
        # assume batch dim = 0
        while s.ndim < x.ndim:
            s = s.view(-1, *([1] * (x.ndim - 1)))
        return x * s

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
        Forward "jump" kernel: q(x_t | x_0) using cumulative alphas:
    
            x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * eps
        """
        # t: (B,) integer timesteps, x: (B, C, ...)
        bar_alpha_t = self.alphas_cumprod[t]          # (B,)
        noise = prior.sample(**prior_kwargs)          # same shape as x
    
        x_jump = self._scale_like(x, bar_alpha_t.sqrt()) + \
                 self._scale_like(noise, (1.0 - bar_alpha_t).sqrt())
    
        return x_jump, noise, None

    def _forward_kernel_prior_noise(self, x_t, t, prior, **prior_kwargs):
        """
        Markov forward kernel using per-step α_t:
    
            x_{t+1} = sqrt(alpha_t) * x_t + sqrt(1 - alpha_t) * eps
        """
        alpha_t = self.alphas[t]                    # α_t = 1 - β_t
        noise = prior.sample(**prior_kwargs)
    
        x_t_next = self._scale_like(x_t, alpha_t.sqrt()) + \
                   self._scale_like(noise, (1.0 - alpha_t).sqrt())
    
        return x_t_next, noise, None
    
    def _forward_kernel_network_noise(self, x_t, t, prior, backbone, network_pred_type,
                                      t_next=None, **kwargs):
        """
        Forward kernel using the network-predicted noise or x0, consistent with the
        training distribution q(x_t | x_0) defined by alphas_cumprod.
    
        If network_pred_type == "noise":
            eps_pred = backbone(x_t, alpha_bar_t)
            x0_t = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
            x_{t_next} = sqrt(alpha_bar_{t_next}) * x0_t + sqrt(1 - alpha_bar_{t_next}) * eps_pred
    
        If network_pred_type == "x0":
            x0_t = backbone(x_t, alpha_bar_t)
            eps_pred = (x_t - sqrt(alpha_bar_t) * x0_t) / sqrt(1 - alpha_bar_t)
        """
        if t_next is None:
            # assume t_next = t+1 if not provided
            t_next = t + 1
    
        bar_alpha_t = self.alphas_cumprod[t]        # (B,)
        bar_alpha_next = self.alphas_cumprod[t_next]
    
        if network_pred_type == "noise":
            eps_pred = backbone(x_t, bar_alpha_t)
            noise_rescaled = self._scale_like(eps_pred, (1.0 - bar_alpha_t).sqrt())
            x0_t = self._scale_like(x_t - noise_rescaled, 1.0 / bar_alpha_t.sqrt())
    
        elif network_pred_type == "x0":
            x0_t = backbone(x_t, bar_alpha_t)
            eps_pred = (x_t - self._scale_like(x0_t, bar_alpha_t.sqrt())) / \
                       (1.0 - bar_alpha_t).sqrt()
    
        else:
            raise ValueError("network_pred_type must be 'noise' or 'x0'")
    
        x_t_next = self._scale_like(x0_t, bar_alpha_next.sqrt()) + \
                   self._scale_like(eps_pred, (1.0 - bar_alpha_next).sqrt())
    
        return x_t_next, eps_pred, None


    def _reverse_kernel(self, x_t, t, backbone, network_pred_type, **kwargs):
        """
        Reverse kernel stepping t -> t-1:
    
            x_prev = sqrt(bar_alpha_{t-1}) * x0_t + sqrt(1 - bar_alpha_{t-1}) * eps_pred
    
        where x0_t and eps_pred are reconstructed from x_t using the same
        bar_alpha_t that defined q(x_t | x_0) in training.
        """
        # t: (B,)
        bar_alpha_t = self.alphas_cumprod[t]                # (B,)
    
        # Define bar_alpha_{t-1}; for t==0 we can treat bar_alpha_{-1} = 1.0
        t_prev = t - 1
        t_prev = torch.clamp(t_prev, min=0)
        bar_alpha_prev = self.alphas_cumprod[t_prev]
    
        if network_pred_type == "noise":
            eps_pred = backbone(x_t, bar_alpha_t)
            noise_rescaled = self._scale_like(eps_pred, (1.0 - bar_alpha_t).sqrt())
            x0_t = self._scale_like(x_t - noise_rescaled, 1.0 / bar_alpha_t.sqrt())
    
        elif network_pred_type == "x0":
            x0_t = backbone(x_t, bar_alpha_t)
            eps_pred = (x_t - self._scale_like(x0_t, bar_alpha_t.sqrt())) / \
                       (1.0 - bar_alpha_t).sqrt()
        else:
            raise ValueError("Please provide a valid prediction type: 'noise' or 'x0'")
    
        x_t_prev = self._scale_like(x0_t, bar_alpha_prev.sqrt()) + \
                   self._scale_like(eps_pred, (1.0 - bar_alpha_prev).sqrt())
    
        return x_t_prev, eps_pred, None


    def _reverse_kernel_jump(self, x_t, t, backbone, network_pred_type, **kwargs):
        """
        Reverse "jump" kernel p(x0 | x_t) used in training:
    
            x0_t = (x_t - sqrt(1 - bar_alpha_t) * eps_pred) / sqrt(bar_alpha_t)
        """
        bar_alpha_t = self.alphas_cumprod[t]
    
        if network_pred_type == "noise":
            eps_pred = backbone(x_t, bar_alpha_t)
            noise_rescaled = self._scale_like(eps_pred, (1.0 - bar_alpha_t).sqrt())
            x0_t = self._scale_like(x_t - noise_rescaled, 1.0 / bar_alpha_t.sqrt())
        elif network_pred_type == "x0":
            x0_t = backbone(x_t, bar_alpha_t)
            eps_pred = (x_t - self._scale_like(x0_t, bar_alpha_t.sqrt())) / \
                       (1.0 - bar_alpha_t).sqrt()
        else:
            raise ValueError("Please provide a valid prediction type: 'noise' or 'x0'")
    
        return x0_t, eps_pred, None


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
            x_next, noise, _ = self.kernel(
                mode="forward",
                x_t=x,
                t=t,
                t_next=t_next,                 # <-- pass t_next through
                prior=prior,
                likelihood=likelihood,
                backbone=backbone,
                network_pred_type=network_pred_type,
                **kwargs
            )
    
        elif mode == "reverse":
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
