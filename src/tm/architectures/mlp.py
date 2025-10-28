import math
import torch
from torch import nn

def exists(x): return x is not None
def default(val, d): return val if exists(val) else (d() if callable(d) else d)

# ----- helpers -----
class LayerNorm1D(nn.Module):
    """LayerNorm over channel dimension for (N, C, L) tensors"""
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        # rearrange to (N, L, C) for nn.LayerNorm
        return self.ln(x.transpose(1,2)).transpose(1,2)

# ----- time embedding -----
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=device) * -(math.log(10000)/ (half - 1)))
        args = t[:, None] * freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# ----- FiLM MLP block -----
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_mult=4, time_emb_dim=None, dropout=0.0):
        super().__init__()
        hidden = dim * hidden_mult
        self.norm1 = LayerNorm1D(dim)
        self.fc1   = nn.Linear(dim, hidden)
        self.act   = nn.SiLU()
        self.do    = nn.Dropout(dropout)
        self.fc2   = nn.Linear(hidden, dim)
        self.norm2 = LayerNorm1D(dim)

        self.film = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim * 2))
            if exists(time_emb_dim) else None
        )

    def forward(self, x, t_emb=None):
        # x: (N, C, L)
        x_ = x.transpose(1,2)  # (N, L, C)
        h = self.norm1(x).transpose(1,2)
        if exists(self.film) and exists(t_emb):
            scale, shift = self.film(t_emb).chunk(2, dim=-1)
            scale = scale[:, None, :]
            shift = shift[:, None, :]
            h = h * (scale + 1) + shift
        h = self.fc1(h)
        h = self.act(h)
        h = self.do(h)
        h = self.fc2(h)
        h = self.norm2(h.transpose(1,2)).transpose(1,2)
        return (x_ + h).transpose(1,2)

# ----- MLP1D pure -----
class MLP(nn.Module):
    """
    Fully-connected MLP backbone for diffusion models.
    No convolutions, same I/O contract as Unet1D.
    """
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        num_resolutions=4,
        channels=3,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        mlp_hidden_mult=4,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        time_dim = dim * 4

        self.input_proj = nn.Linear(input_channels, init_dim)

        sinu_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        depth = 2 * num_resolutions + 2
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(init_dim, mlp_hidden_mult, time_emb_dim=time_dim, dropout=dropout)
            for _ in range(depth)
        ])

        default_out_dim = channels * (2 if learned_variance else 1)
        self.out_proj = nn.Linear(init_dim, default_out_dim)

    def forward(self, x, time, x_self_cond=None):
        """
        x: (N, C, L)
        time: (N,)
        return: (N, out_dim, L)
        """
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat([x_self_cond, x], dim=1)

        # project across channel dimension
        x = x.transpose(1,2)  # (N, L, C_in)
        x = self.input_proj(x)  # (N, L, dim)
        t_emb = self.time_mlp(time)

        for blk in self.blocks:
            x = blk(x.transpose(1,2), t_emb).transpose(1,2)

        x = self.out_proj(x)  # (N, L, out_dim)
        return x.transpose(1,2)
