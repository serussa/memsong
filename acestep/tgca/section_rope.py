"""
SectionRoPEOffset — V2: head-specific K-side RoPE phase offset for cross-attention.

V2 changes vs V1:
  - ``section_phase`` is [num_heads, num_section_types, rope_pair_dim] (was [N, D])
  - Per-head ``log_scale`` [num_heads, 1, 1] (was scalar)
  - Optional token-aware phase weighting
  - Tighter default clamp (0.03 rad vs 0.2)
  - UNKNOWN=0 always zero'd

Only affects K side, first ``time_dim`` dimensions, only at specified layers.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section type vocabulary (matches lyrics_parser.py)
# ---------------------------------------------------------------------------
SECTION_UNKNOWN = 0
SECTION_INTRO = 1
SECTION_VERSE = 2
SECTION_PRECHORUS = 3
SECTION_CHORUS = 4
SECTION_BRIDGE = 5
SECTION_OUTRO = 6
SECTION_INSTRUMENTAL = 7
NUM_SECTION_TYPES = 8


class SectionRoPEOffset(nn.Module):
    """Head-specific section-type phase offset for RoPE.

    For each attention head, maintains a per-section-type phase delta
    applied to the first ``rope_pair_dim`` frequency pairs (i.e. the first
    2 * rope_pair_dim = ``time_dim`` dimensions) of the K states.

    Initialized at zero so the module is a no-op at start.
    """

    def __init__(
        self,
        num_heads: int = 16,
        num_section_types: int = NUM_SECTION_TYPES,
        rope_pair_dim: int = 16,
        max_offset: float = 0.03,
        init_log_scale: float = -3.5,
        strength: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.rope_pair_dim = rope_pair_dim
        self.max_offset = max_offset
        self.strength = strength

        # [num_heads, num_sections, rope_pair_dim]
        self.section_phase = nn.Parameter(
            torch.zeros(num_heads, num_section_types, rope_pair_dim)
        )
        # Per-head log scale: [num_heads, 1, 1]
        self.log_scale = nn.Parameter(
            torch.full((num_heads, 1, 1), init_log_scale)
        )

    def forward(
        self,
        section_ids: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute section-conditioned head-specific phase offset.

        Args:
            section_ids: [B, L] integer section type IDs (0=UNKNOWN … 7).
            token_weights: [B, L] or None.  Optional per-token scaling.

        Returns:
            delta: [B, H, L, rope_pair_dim] phase offset in radians.
                   Clamped to [-max_offset, max_offset].
        """
        B, L = section_ids.shape
        H, N, D = self.section_phase.shape  # H=num_heads, N=num_sections, D=rope_pair_dim

        # Embedding lookup: treat [H, N, D] as N entries of H*D-dim vectors
        w = self.section_phase.permute(1, 0, 2).reshape(N, H * D)  # [N, H*D]
        delta = F.embedding(section_ids, w)  # [B, L, H*D]
        delta = delta.view(B, L, H, D).permute(0, 2, 1, 3)  # [B, H, L, D]

        # Per-head scale
        scale = self.log_scale.exp()  # [H, 1, 1]
        delta = delta * scale

        # Store for diagnostics
        self._last_token_weights = token_weights

        # Token-aware phase weighting
        if token_weights is not None:
            delta = delta * token_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]

        # Global strength multiplier
        if self.strength != 1.0:
            delta = delta * self.strength

        # Clamp
        delta = torch.clamp(delta, -self.max_offset, self.max_offset)

        # UNKNOWN (ID=0) no-op: section type 0 always produces delta 0
        delta = delta * (section_ids != SECTION_UNKNOWN).float().view(B, 1, L, 1)

        return delta


# ---------------------------------------------------------------------------
# RoPE with head-specific phase offset
# ---------------------------------------------------------------------------

def rope_with_phase_offset(
    x: torch.Tensor,
    time_dim: int,
    phase_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply per-head phase rotation to the first *time_dim* dims of *x*.

    When *phase_offset* is [B, H, L, Dp], rotates K per-head in the complex
    plane by the head-specific section-conditioned phase angle.

    Args:
        x: [B, H, L, Dh] — tensor to rotate (typically K).
        time_dim: Dt — number of dims to rotate (must be ≤ x.size(-1), even).
        phase_offset: [B, H, L, Dp] or None — per-head, per-token phase angles.

    Returns:
        x_rotated: [B, H, L, Dh] — first *time_dim* dims rotated, rest unchanged.
    """
    if phase_offset is None:
        return x

    # Cast phase_offset to match x's dtype if needed (prevents dtype mismatch in attention)
    if phase_offset.dtype != x.dtype:
        phase_offset = phase_offset.to(x.dtype)

    x_time = x[..., :time_dim]
    x_rest = x[..., time_dim:]

    Dt = x_time.shape[-1]
    Dp = Dt // 2

    delta = phase_offset[..., :Dp]      # [B, H, L, Dp]
    emb = torch.cat([delta, delta], dim=-1)  # [B, H, L, Dt]

    cos = emb.cos()
    sin = emb.sin()

    x1 = x_time[..., 0::2]    # even-indexed pairs
    x2 = x_time[..., 1::2]    # odd-indexed pairs

    y1 = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
    y2 = x1 * sin[..., 0::2] + x2 * cos[..., 0::2]

    y = torch.stack([y1, y2], dim=-1).flatten(-2)

    return torch.cat([y, x_rest], dim=-1)
