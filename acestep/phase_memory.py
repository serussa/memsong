"""PhaseMemory module and training helpers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseMemory(nn.Module):
    """
    A. Closed-loop entropy (state + step)
    B. Adaptive anchor (state-conditioned slow manifold)
    C. Trajectory memory (actually used)
    """

    def __init__(self, dim, mem_dim=128):
        super().__init__()

        self.mem_dim = mem_dim

        self.proj_r = nn.Linear(dim, mem_dim)
        self.proj_i = nn.Linear(dim, mem_dim)

        self.omega = nn.Linear(dim + 3 * mem_dim, mem_dim)

        # entropy controller
        self.entropy_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # adaptive anchor conditioned update
        self.anchor_net = nn.Linear(mem_dim * 2, mem_dim)

        self.out = nn.Linear(mem_dim * 2, dim)
        nn.init.zeros_(self.out.weight)

        self.register_buffer("z_r", None)
        self.register_buffer("z_i", None)
        self.register_buffer("traj", None)
        self.register_buffer("anchor", torch.zeros(1, 1, mem_dim))

    def forward(self, h, diffusion_step):

        B, T, D = h.shape

        # -------------------
        # complex projection
        # -------------------
        zr = self.proj_r(h)
        zi = self.proj_i(h)

        norm = torch.sqrt(zr**2 + zi**2 + 1e-6)
        zr = zr / norm
        zi = zi / norm

        # -------------------
        # init/reset memory
        # -------------------
        if self.z_r is None or self.z_r.shape[:2] != (B, T):
            self.z_r = zr.detach()
            self.z_i = zi.detach()
            self.traj = None

        # -------------------
        # closed-loop entropy (FIXED)
        # -------------------
        step_ratio = diffusion_step.to(zr.dtype).view(B, 1, 1) / 1000.0
        if step_ratio.shape[1] != T:
            step_ratio = step_ratio.expand(B, T, 1)

        energy = (zr**2 + zi**2).mean(dim=-1, keepdim=True)
        drift = (zr - self.z_r).abs().mean(dim=-1, keepdim=True)

        ctrl = torch.cat([step_ratio, energy, drift], dim=-1)
        ctrl = ctrl.to(self.entropy_net[0].weight.dtype)

        alpha = torch.sigmoid(self.entropy_net(ctrl))  # [B,T,1]

        # -------------------
        # adaptive anchor (STATE-DEPENDENT)
        # -------------------
        anchor_input = torch.cat([self.z_r, self.z_i], dim=-1)
        anchor_update = self.anchor_net(anchor_input).mean(dim=(0, 1), keepdim=True)

        self.anchor = 0.995 * self.anchor + 0.005 * anchor_update.detach()

        anchor = self.anchor.expand(B, T, -1)

        # -------------------
        # phase dynamics
        # -------------------
        omega_in = torch.cat([h, zr, zi, anchor], dim=-1)
        omega = math.pi * torch.tanh(self.omega(omega_in))

        c = torch.cos(omega)
        s = torch.sin(omega)

        zr_rot = zr * c - zi * s
        zi_rot = zr * s + zi * c

        # -------------------
        # update
        # -------------------
        zr_new = zr_rot + alpha * (zr + anchor)
        zi_new = zi_rot + alpha * (zi + anchor)

        # -------------------
        # trajectory memory (USED)
        # -------------------
        traj = torch.cat([zr_new, zi_new], dim=-1)

        if self.traj is None:
            self.traj = traj.detach()

        self.traj = 0.9 * self.traj + 0.1 * traj.detach()

        # optional: feed back (IMPORTANT)
        zr_new = zr_new + 0.1 * self.traj[..., :self.mem_dim]
        zi_new = zi_new + 0.1 * self.traj[..., self.mem_dim:]

        # -------------------
        # stabilize
        # -------------------
        scale = torch.sqrt(zr_new**2 + zi_new**2 + 1.0)
        zr_new = zr_new / scale
        zi_new = zi_new / scale

        # -------------------
        # persist
        # -------------------
        self.z_r = zr_new.detach()
        self.z_i = zi_new.detach()

        z = torch.cat([zr_new, zi_new], dim=-1)
        return h + self.out(z)

    def reset(self) -> None:
        """Reset persistent diffusion-phase state for a fresh inference run."""
        self.z_r = None
        self.z_i = None
        self.traj = None
        self.anchor.zero_()


def freeze_except_phase_memory(model: nn.Module) -> tuple[int, int]:
    """Freeze all parameters except those in PhaseMemory modules.

    Args:
        model: Model containing PhaseMemory sub-modules.

    Returns:
        tuple[int, int]: (trainable_params, total_params).
    """
    phase_memory_params: set[int] = set()
    for module in model.modules():
        if isinstance(module, PhaseMemory):
            for param in module.parameters():
                phase_memory_params.add(id(param))

    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if id(param) in phase_memory_params:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    return trainable_params, total_params


def unfreeze_all(model: nn.Module) -> None:
    """Enable gradient computation for all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def reset_phase_memory(model: nn.Module) -> None:
    """Reset all PhaseMemory modules inside a model."""
    for module in model.modules():
        if isinstance(module, PhaseMemory):
            module.reset()