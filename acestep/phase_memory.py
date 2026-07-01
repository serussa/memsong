"""
PhaseMemory v2 — PhaseMemory-Conditioned Duration Clock (PMDC).

Replaces old PhaseMemory (hidden/KV injector) with a **progress-only
duration clock**.  PMDC reads hidden states and outputs:

  - ``s_pm``: bounded log-speed residual  [B, T]
  - ``p_final``: monotonic progress in [0, 1]  [B, T]

Additionally provides:
  - ``LyricUnit`` dataclass for structured lyric parsing
  - ``build_duration_scaffold`` to construct temporal boundaries
  - ``build_duration_interval_bias`` for cross-attention logit shaping
  - ``mass_preserving_attention`` with split-softmax
  - ``pmdc_regularization_loss`` (optional)

Usage (forward pass)::

    clock = PhaseMemoryDurationClock(dim=2048)
    s_pm, p_final = clock(h, p_base)

    scaffold = build_duration_scaffold(units, text_len, device=h.device)
    bias = build_duration_interval_bias(p_final, scaffold["unit_boundaries"],
                                        scaffold["token_to_unit"], scaffold["lyric_mask"])
    out, attn_new, stats = mass_preserving_attention(logits, value,
                                                      lyric_mask=scaffold["lyric_mask"],
                                                      bias=bias, gate=0.1)
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================================================
#  PhaseMemoryDurationClock
# ===================================================================


class PhaseMemoryDurationClock(nn.Module):
    """Progress-only recurrent clock driven by hidden-state dynamics.

    Reads ``h`` (and optionally its temporal difference) and outputs a
    bounded log-speed residual ``s_pm``.  When composed with a base
    progress schedule, this produces a monotonic warped progress
    ``p_final`` that can drive duration-interval attention biases.

    Key constraints (enforced by design):
      - No hidden residual / K/V modification / trajectory / anchor.
      - Zero-initialised final layer so ``p_final ≈ p_base`` at init.
      - Per-step state normalisation prevents numerical drift.
    """

    def __init__(
        self,
        dim: int,
        mem_dim: int = 128,
        hidden_dim: int = 256,
        beta: float = 0.1,
        use_delta_h: bool = True,
    ):
        super().__init__()
        self.mem_dim = mem_dim
        self.beta = beta
        self.use_delta_h = use_delta_h

        in_dim = dim * 2 if use_delta_h else dim

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )

        self.proj_r = nn.Linear(hidden_dim, mem_dim)
        self.proj_i = nn.Linear(hidden_dim, mem_dim)

        self.omega = nn.Linear(hidden_dim + 2 * mem_dim, mem_dim)

        self.speed_head = nn.Sequential(
            nn.LayerNorm(2 * mem_dim),
            nn.Linear(2 * mem_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Zero init so initial s_pm ≈ 0 → p_final ≈ p_base
        nn.init.zeros_(self.speed_head[-1].weight)
        nn.init.zeros_(self.speed_head[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        p_base: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, T, D] hidden states.
            p_base: [T] or [B, T] or None.  If None, linear [0, 1].

        Returns:
            s_pm: [B, T] bounded log-speed residual.
            p_final: [B, T] monotonic progress in [0, 1].
        """
        B, T, _ = h.shape
        device, dtype = h.device, h.dtype

        # ---- build input features -------------------------------------------
        if self.use_delta_h:
            delta_h = torch.zeros_like(h)
            delta_h[:, 1:] = h[:, 1:] - h[:, :-1]
            x = torch.cat([h, delta_h], dim=-1)  # [B, T, 2D]
        else:
            x = h

        x = self.input_proj(x)  # [B, T, hidden_dim]

        # ---- base progress schedule -----------------------------------------
        if p_base is None:
            p_base = torch.linspace(0, 1, T, device=device, dtype=dtype)
            p_base = p_base.unsqueeze(0).expand(B, T)
        elif p_base.dim() == 1:
            p_base = p_base.unsqueeze(0).expand(B, T)
        else:
            assert p_base.shape == (B, T), f"p_base shape {p_base.shape} != ({B}, {T})"

        # base speed
        v_base = torch.zeros_like(p_base)
        v_base[:, 0] = p_base[:, 1] - p_base[:, 0] if T > 1 else 1.0
        v_base[:, 1:] = p_base[:, 1:] - p_base[:, :-1]
        v_base = torch.clamp(v_base, min=1e-5)
        v_base = v_base / (v_base.mean(dim=1, keepdim=True) + 1e-6)
        log_v_base = torch.log(v_base)

        # ---- recurrent clock -------------------------------------------------
        zr = torch.zeros(B, self.mem_dim, device=device, dtype=dtype)
        zi = torch.zeros(B, self.mem_dim, device=device, dtype=dtype)

        s_pm_list = []
        for t in range(T):
            xt = x[:, t]  # [B, hidden_dim]

            inp_r = self.proj_r(xt)
            inp_i = self.proj_i(xt)

            omega_in = torch.cat([xt, zr, zi], dim=-1)
            omega = math.pi * torch.tanh(self.omega(omega_in))

            c = torch.cos(omega)
            s = torch.sin(omega)

            zr_new = zr * c - zi * s + inp_r
            zi_new = zr * s + zi * c + inp_i

            # Normalise to prevent drift
            scale = torch.sqrt(zr_new ** 2 + zi_new ** 2 + 1.0)
            zr = zr_new / scale
            zi = zi_new / scale

            state = torch.cat([zr, zi], dim=-1)  # [B, 2*mem_dim]
            s_t = self.speed_head(state).squeeze(-1)  # [B]
            s_pm_list.append(s_t)

        s_pm = torch.stack(s_pm_list, dim=1)  # [B, T]

        # ---- compose speed & integrate ---------------------------------------
        log_v = log_v_base + self.beta * torch.tanh(s_pm)
        v = torch.exp(log_v)

        p_final = torch.cumsum(v, dim=1)
        p_final = (p_final - p_final[:, :1]) / (p_final[:, -1:] - p_final[:, :1] + 1e-6)
        p_final = torch.clamp(p_final, 0.0, 1.0)

        return s_pm, p_final


# ===================================================================
#  PMDCResidualClock — trainable residual lyric clock
# ===================================================================


class PMDCResidualClock(nn.Module):
    """Trainable progress-speed residual clock for PMDC.

    A lightweight MLP that reads per-timestep hidden states and outputs
    a bounded log-speed residual.  Composed with a base linear progress
    schedule ``p_base``, this produces a warped monotonic progress
    ``p_final`` that drives duration-interval attention bias.

    Design:
      - Non-recurrent (per-timestep MLP) — simpler than
        ``PhaseMemoryDurationClock``, easier to train.
      - Zero-initialised final layer → ``p_final ≈ p_base`` at init.
      - ``beta`` is a learnable scalar (clipped to ``[0, beta_max]``).
      - ``use_delta_h=True`` appends the temporal difference of hidden
        states as additional input features.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        beta_init: float = 0.05,
        beta_max: float = 0.15,
        use_delta_h: bool = True,
    ):
        super().__init__()
        self.beta_max = beta_max
        self.use_delta_h = use_delta_h

        in_dim = dim * 2 if use_delta_h else dim

        self.input_norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Zero init → initial speed_residual ≈ 0 → p_final ≈ p_base
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Learnable beta (clamped to [0, beta_max])
        self.beta_logit = nn.Parameter(torch.tensor(0.0))
        # Initialise so beta ≈ beta_init
        self._reset_beta(beta_init)

    def _reset_beta(self, beta_init: float):
        # We want beta ≈ beta_init, so: sigmoid(logit) * beta_max = beta_init
        # logit = log(beta_init / (beta_max - beta_init))
        eps = 1e-8
        ratio = max(beta_init / max(self.beta_max, eps), eps)
        ratio = min(ratio, 1.0 - eps)  # clamp away from 1
        init_logit = math.log(ratio / (1.0 - ratio))
        with torch.no_grad():
            self.beta_logit.fill_(init_logit)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_logit) * self.beta_max

    def forward(
        self,
        h: torch.Tensor,
        p_base: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, T, D] hidden states.
            p_base: [T] or [B, T] or None.  If None, linear [0, 1].

        Returns:
            speed_residual: [B, T] bounded log-speed residual.
            p_final: [B, T] monotonic progress in [0, 1].
            log_v: [B, T] log-speed (for smoothness loss).
        """
        B, T, _ = h.shape
        device, dtype = h.device, h.dtype

        # ---- input features --------------------------------------------------
        if self.use_delta_h:
            delta_h = torch.zeros_like(h)
            delta_h[:, 1:] = h[:, 1:] - h[:, :-1]
            x = torch.cat([h, delta_h], dim=-1)
        else:
            x = h

        x = self.input_norm(x)  # [B, T, in_dim]
        speed_residual = self.net(x).squeeze(-1)  # [B, T]

        # ---- base progress schedule ------------------------------------------
        if p_base is None:
            p_base = torch.linspace(0, 1, T, device=device, dtype=dtype)
            p_base = p_base.unsqueeze(0).expand(B, T)
        elif p_base.dim() == 1:
            p_base = p_base.unsqueeze(0).expand(B, T)

        # base speed
        v_base = torch.zeros_like(p_base)
        v_base[:, 0] = p_base[:, 1] - p_base[:, 0] if T > 1 else 1.0
        v_base[:, 1:] = p_base[:, 1:] - p_base[:, :-1]
        v_base = torch.clamp(v_base, min=1e-5)
        v_base = v_base / (v_base.mean(dim=1, keepdim=True) + 1e-6)
        log_v_base = torch.log(v_base)

        # ---- compose speed & integrate ---------------------------------------
        beta_val = self.beta  # scalar
        log_v = log_v_base + beta_val * torch.tanh(speed_residual)
        v = torch.exp(log_v)

        p_final = torch.cumsum(v, dim=1)
        p_final = (p_final - p_final[:, :1]) / (p_final[:, -1:] - p_final[:, :1] + 1e-6)
        p_final = torch.clamp(p_final, 0.0, 1.0)

        return speed_residual, p_final, log_v


# ===================================================================
#  PhaseControlledLyricScaffoldPM — the new trainable PM
# ===================================================================

class PhaseControlledLyricScaffoldPM(nn.Module):
    """Phase-Controlled Lyric Scaffold PM.

    Takes hidden states + static p_base anchor → produces:
      - p_dyn: warped monotonic progress [B, T]
      - pm_hidden_residual: hidden-state residual [B, T, D]

    p_dyn drives:
      1. duration-interval bias for cross-attention
      2. conditions the hidden residual head

    The hidden residual is injected back into the layer-12 hidden
    (via forward hook during training), so flow_loss gradients
    flow through remaining 12 DiT layers → PM parameters.
    """

    def __init__(
        self,
        dim: int,
        mem_dim: int = 128,
        coord_dim: int = 32,
        hidden_dim: int = 256,
        beta_max: float = 0.15,
    ):
        super().__init__()
        self.dim = dim
        self.mem_dim = mem_dim
        self.beta_max = beta_max

        # Coordinate MLP: p_base features → coord embedding
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
            nn.SiLU(),
        )

        # Input: h + dh + coord_emb
        in_dim = dim * 2 + coord_dim
        self.input_norm = nn.LayerNorm(in_dim)

        # Phase trunk (recurrent, like old PM)
        self.proj_r = nn.Linear(hidden_dim, mem_dim)
        self.proj_i = nn.Linear(hidden_dim, mem_dim)
        self.omega = nn.Linear(hidden_dim + 2 * mem_dim, mem_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )

        # Warp head: PM state → speed residual
        self.warp_head = nn.Sequential(
            nn.LayerNorm(2 * mem_dim),
            nn.Linear(2 * mem_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Dyn coord MLP: p_dyn, p_base, residual → embedding
        self.dyn_coord_mlp = nn.Sequential(
            nn.Linear(3, coord_dim),
            nn.SiLU(),
        )

        # Residual head: PM state + h + dh + dyn_coord → hidden residual
        residual_in_dim = 2 * mem_dim + dim * 2 + coord_dim
        self.residual_head = nn.Sequential(
            nn.LayerNorm(residual_in_dim),
            nn.Linear(residual_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

        # Zero-init critical heads
        nn.init.zeros_(self.warp_head[-1].weight)
        nn.init.zeros_(self.warp_head[-1].bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        # Learnable gates
        self.beta_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit) * self.beta_max

    def forward(
        self,
        h: torch.Tensor,
        p_base: torch.Tensor,
        output_residual: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, T, D] hidden states (from layer 12 or any layer).
            p_base: [B, T] normalised progress anchor in [0, 1].
            output_residual: If True, also compute and return hidden residual.

        Returns:
            p_dyn: [B, T] warped monotonic progress.
            speed_residual: [B, T] bounded log-speed residual.
            pm_hidden_residual: [B, T, D] hidden residual, zero when
                output_residual=False.
        """
        B, T, D = h.shape
        device, dtype = h.device, h.dtype

        # ---- Input features ---------------------------------------------------
        # h_avg = h.mean(dim=1, keepdim=True)  # [B, 1, D]
        # delta_h
        dh = torch.zeros_like(h)
        dh[:, 1:] = h[:, 1:] - h[:, :-1]

        # coord embedding from p_base
        p_exp = p_base.unsqueeze(-1)  # [B, T, 1]
        coord_feat = torch.cat([
            p_exp,
            p_exp ** 2,
            1.0 - p_exp,
        ], dim=-1)  # [B, T, 3]
        coord_emb = self.coord_mlp(coord_feat)  # [B, T, coord_dim]

        # Concatenate input
        x = torch.cat([h, dh, coord_emb], dim=-1)  # [B, T, 2*D + coord_dim]
        x = self.input_norm(x)
        x = self.trunk(x)  # [B, T, hidden_dim]

        # ---- Phase trunk (recurrent) ------------------------------------------
        zr = torch.zeros(B, self.mem_dim, device=device, dtype=dtype)
        zi = torch.zeros(B, self.mem_dim, device=device, dtype=dtype)
        all_zr = []
        all_zi = []

        for t in range(T):
            xt = x[:, t]  # [B, hidden_dim]
            inp_r = self.proj_r(xt)
            inp_i = self.proj_i(xt)
            omega_in = torch.cat([xt, zr, zi], dim=-1)
            omega = math.pi * torch.tanh(self.omega(omega_in))
            c = torch.cos(omega)
            s = torch.sin(omega)
            zr_new = zr * c - zi * s + inp_r
            zi_new = zr * s + zi * c + inp_i
            scale = torch.sqrt(zr_new ** 2 + zi_new ** 2 + 1.0)
            zr = zr_new / scale
            zi = zi_new / scale
            all_zr.append(zr)
            all_zi.append(zi)

        zr_stack = torch.stack(all_zr, dim=1)   # [B, T, mem_dim]
        zi_stack = torch.stack(all_zi, dim=1)   # [B, T, mem_dim]
        pm_state = torch.cat([zr_stack, zi_stack], dim=-1)  # [B, T, 2*mem_dim]

        # ---- Warp head: PM state → speed residual → p_dyn --------------------
        speed_residual = self.warp_head(pm_state).squeeze(-1)  # [B, T]

        # Build p_dyn from p_base + speed residual
        v_base = torch.zeros_like(p_base)
        v_base[:, 0] = p_base[:, 1] - p_base[:, 0] if T > 1 else 1.0
        v_base[:, 1:] = p_base[:, 1:] - p_base[:, :-1]
        v_base = torch.clamp(v_base, min=1e-5)
        v_base = v_base / (v_base.mean(dim=1, keepdim=True) + 1e-6)
        log_v_base = torch.log(v_base)

        beta_val = self.beta  # scalar
        log_v = log_v_base + beta_val * torch.tanh(speed_residual)
        v = torch.exp(log_v)

        p_dyn = torch.cumsum(v, dim=1)
        p_dyn = (p_dyn - p_dyn[:, :1]) / (p_dyn[:, -1:] - p_dyn[:, :1] + 1e-6)
        p_dyn = torch.clamp(p_dyn, 0.0, 1.0)

        # ---- Hidden residual (if requested) -----------------------------------
        pm_hidden_residual = torch.zeros_like(h)
        if output_residual and T > 0:
            dyn_coord_feat = torch.stack([
                p_dyn,
                p_base,
                p_dyn - p_base,
            ], dim=-1)  # [B, T, 3]
            dyn_coord_emb = self.dyn_coord_mlp(dyn_coord_feat)  # [B, T, coord_dim]

            residual_in = torch.cat([
                pm_state,          # [B, T, 2*mem_dim]
                h,                 # [B, T, D]
                dh,                # [B, T, D]
                dyn_coord_emb,     # [B, T, coord_dim]
            ], dim=-1)
            pm_hidden_residual = self.residual_head(residual_in)  # [B, T, D]

        return p_dyn, speed_residual, pm_hidden_residual


# ===================================================================
#  LyricUnit
# ===================================================================


@dataclass
class LyricUnit:
    """A single structural unit in the lyric scaffold."""

    unit_id: int
    section: str
    text: str
    char_count: int = 0
    token_indices: List[int] = field(default_factory=list)
    occurrence_id: int = 0
    is_silence: bool = False
    is_control: bool = False
    """True for natural control lines such as ``[Intro - Instrumental]``
    that have real tokeniser tokens and serve as non-lyric attention targets."""
    is_lyric: bool = True
    """True for real sung-lyric lines.  A unit with ``is_control=False`` and
    ``is_silence=False`` is a lyric unit."""
    duration_weight: float = 0.0
    """If > 0, overrides the auto-computed weight in ``build_duration_scaffold``.
    Used by auto-inserted silence units (intro, outro, transitions) to achieve
    a target duration ratio."""


# ===================================================================
#  Duration scaffold
# ===================================================================

SECTION_MULTIPLIER = {
    "INTRO": 1.00,
    "VERSE": 1.00,
    "PRECHORUS": 1.05,
    "PRE-CHORUS": 1.05,
    "CHORUS": 1.15,
    "BRIDGE": 1.05,
    "OUTRO": 1.00,
    "INSTRUMENTAL": 1.00,
    "INSTR": 1.00,
    "UNKNOWN": 1.00,
}

# ---------------------------------------------------------------------------
# Section tag mapping
# ---------------------------------------------------------------------------

SECTION_TAG_MAP: Dict[str, str] = {
    "INTRO": "INTRO",
    "VERSE": "VERSE",
    "PRECHORUS": "PRECHORUS",
    "PRE-CHORUS": "PRECHORUS",
    "CHORUS": "CHORUS",
    "BRIDGE": "BRIDGE",
    "INTERLUDE": "INTERLUDE",
    "INSTRUMENTAL": "INSTRUMENTAL",
    "INSTR": "INSTRUMENTAL",
    "SOLO": "SOLO",
    "OUTRO": "OUTRO",
    "INST": "INSTRUMENTAL",
}

NON_LYRIC_SECTIONS = {"INTRO", "INTERLUDE", "INSTRUMENTAL", "SOLO", "OUTRO"}
"""Sections that produce silence / instrumental units (no singing)."""

LYRIC_SECTIONS = {"VERSE", "PRECHORUS", "PRE-CHORUS", "CHORUS", "BRIDGE"}
"""Sections that contain sung lyrics."""

# Default auto-transition silence ratios (fraction of total duration)
DEFAULT_AUTO_TRANSITIONS: Dict[str, float] = {
    "intro": 0.06,         # beginning silence
    "chorus_to_verse": 0.04,
    "chorus_to_bridge": 0.03,
    "bridge_to_chorus": 0.03,
    "outro": 0.06,         # ending silence
}
# Maximum total silence ratio
MAX_SILENCE_RATIO = 0.35

# ---------------------------------------------------------------------------
# Natural control line detection
# ---------------------------------------------------------------------------

CONTROL_KEYWORDS = {
    "intro", "outro", "instrumental", "break", "interlude", "solo",
    "fade out", "no vocals", "music only",
    "synth", "piano", "saxophone", "guitar solo", "drum machine",
    "organ", "strings", "pad", "ambient", "atmospheric",
}
"""Keywords that mark a ``[bracketed]`` line as a natural control line
(rather than a lyric section tag)."""


def is_natural_control_line(line: str) -> bool:
    """Check if a bracketed line is a natural control description.

    Must contain additional descriptive text beyond just the bare keyword —
    bare ``[Intro]``, ``[Verse]`` etc. are section tags, not control lines.

    ``[Verse]`` → False (lyric section tag)
    ``[Intro - Instrumental]`` → True (natural control)
    ``[Outro - Piano Fade Out]`` → True
    """
    m = re.match(r"^\[(.+)\]$", line.strip())
    if not m:
        return False
    body = m.group(1).strip()
    body_lower = body.lower()

    # If it's a raw section tag (single word, no separator), it's NOT a control line
    # Check for a dash, colon, or multi-word description
    if not any(sep in body for sep in ("-", ":", "—")):
        # Single-word bracket like [Intro], [Verse] — not a control line
        return False

    # Check if any control keyword appears in the body
    for kw in CONTROL_KEYWORDS:
        if kw in body_lower:
            return True
    return False


# ===================================================================
#  Text preprocessing — natural control line insertion
# ===================================================================

def make_control_tag(style: str = "intro"):
    """Generate a natural control line in ACE-Step style.

    Args:
        style: ``intro``, ``outro``, or ``break``.

    Returns:
        Control line string such as ``[Intro - Instrumental]``.
    """
    if style == "intro":
        return "[Intro - Instrumental]"
    elif style == "outro":
        return "[Outro - Instrumental Fade Out]"
    elif style == "break":
        return "[Instrumental Break]"
    return ""


def insert_control_lines(
    lyrics_text: str,
    intro_ratio: float = 0.025,
    outro_ratio: float = 0.035,
    auto_insert_breaks: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Insert natural control lines into raw lyrics text before tokenization.

    The lines are inserted into the text that will be tokenised by the
    Qwen text encoder, so they get real embeddings — not random tokens.

    Args:
        lyrics_text: Raw lyrics string.
        intro_ratio: Target intro duration fraction (for duration scaffold).
        outro_ratio: Target outro duration fraction.
        auto_insert_breaks: If True, insert ``[Instrumental Break]`` at
            CHORUS→VERSE / CHORUS→BRIDGE transitions (default off).

    Returns:
        modified_text: Lyrics with control lines inserted.
        info: Dict with ``intro_inserted``, ``outro_inserted``, ``breaks_inserted``.
    """
    lines = [l for l in lyrics_text.strip().split("\n")]

    # Remove empty trailing/leading
    info: Dict[str, Any] = {"intro_inserted": False, "outro_inserted": False,
                            "breaks_inserted": 0}

    # Detect existing control lines
    has_intro = any(is_natural_control_line(l) and "intro" in l.lower() for l in lines)
    has_outro = any(is_natural_control_line(l) and "outro" in l.lower() for l in lines)

    # Insert intro at beginning
    if not has_intro and intro_ratio > 0:
        lines.insert(0, make_control_tag("intro"))
        info["intro_inserted"] = True

    # Insert outro at end
    if not has_outro and outro_ratio > 0:
        lines.append(make_control_tag("outro"))
        info["outro_inserted"] = True

    # Optional auto-insert breaks at CHORUS→VERSE transitions (off by default)
    if auto_insert_breaks:
        current_section = None
        new_lines = []
        for line in lines:
            m = re.match(r"^\[(.+)\]$", line.strip())
            if m:
                tag = m.group(1).upper().replace("-", "").replace(" ", "")
                if tag in ("CHORUS",) and current_section == "VERSE":
                    new_lines.append(make_control_tag("break"))
                    info["breaks_inserted"] += 1
                current_section = tag if tag in {s.upper() for s in SECTION_TAG_MAP.values()} else current_section
            new_lines.append(line)
        # Avoid duplicate at end
        lines = new_lines

    return "\n".join(lines), info


# ===================================================================
#  Denoising gate fadeout
# ===================================================================

def get_scheduled_gate(
    base_gate: float,
    denoise_progress: float,
    fadeout_start: float = 0.60,
    fadeout_end: float = 0.85,
    final_gate: float = 0.05,
) -> float:
    """Schedule gate strength over denoising progress.

    0.00–0.60: gate = base_gate (full control)
    0.60–0.85: linear decay to final_gate
    0.85–1.00: gate = final_gate (minimal control)

    Args:
        base_gate: Gate value at early denoising steps.
        denoise_progress: Current progress in [0, 1] (0 = noise, 1 = clean).
        fadeout_start: Progress at which to start fading out.
        fadeout_end: Progress at which fadeout completes.
        final_gate: Gate value after fadeout.

    Returns:
        Scheduled gate value.
    """
    if denoise_progress <= fadeout_start:
        return base_gate
    elif denoise_progress >= fadeout_end:
        return final_gate
    else:
        alpha = (denoise_progress - fadeout_start) / (fadeout_end - fadeout_start)
        return base_gate + (final_gate - base_gate) * alpha


# ===================================================================
#  Lyrics parsing — structured tag-aware unit builder
# ===================================================================

def parse_lyrics_to_units(
    lyrics_text: str,
    section_ids: torch.Tensor,
    auto_transition_ratios: Optional[Dict[str, float]] = None,
) -> Tuple[List[LyricUnit], np.ndarray, Dict[str, Any]]:
    """Parse lyrics text into structured LyricUnits with section-tag awareness.

    **Key changes vs the old implementation:**

    1. Section tags (``[Verse]``, ``[Chorus]``, …) only update ``current_section``
       and are **not** treated as lyric units.
    2. Non-singing sections (Intro, Interlude, Instrumental, Solo, Outro) produce
       silence units (``is_silence=True``).
    3. Real lyric lines become lyric units with correct section, occurrence_id,
       and token indices that exclude section-tag tokens.
    4. Section switching is immediate: ``[Chorus]`` → next lyric line is Chorus.
    5. Token spans are estimated via character-length proportion (since the raw
       tokeniser is unavailable here).
    6. Auto-transition silence units are inserted at section boundaries
       (e.g. CHORUS→VERSE, beginning, end).

    Args:
        lyrics_text: Raw lyrics string (with ``[Section]`` tags).
        section_ids: [L] tensor of per-token section IDs (0-7) from
            ``LyricsStructureParser``.  Used for validation and fallback.
        auto_transition_ratios: Override silence ratios.  Keys: ``intro``,
            ``chorus_to_verse``, ``chorus_to_bridge``, ``bridge_to_chorus``,
            ``outro``.  Values in [0, 1].

    Returns:
        units: List of LyricUnit.
        token_pos: [L] position array (-1 = non-lyric, other = normalised in [0,1]).
        debug_info: dict with parsing statistics.
    """
    ratios = {**DEFAULT_AUTO_TRANSITIONS, **(auto_transition_ratios or {})}

    # ── 1. Segment the text into structured lines ────────────────────────────
    raw_lines = lyrics_text.strip().split("\n")
    L = len(section_ids)
    ids_np = section_ids.cpu().numpy() if isinstance(section_ids, torch.Tensor) else section_ids

    segments: List[Tuple[str, str, str, int]] = []
    """(type, content, section, char_count). type is 'tag' or 'lyric'."""

    current_section = "UNKNOWN"
    tag_line_count = 0
    lyric_line_count = 0

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Detect section tag or control line: [Something]
        m = re.match(r"^\[(.+)\]$", stripped)
        if m:
            tag_raw = m.group(1).strip().upper()
            mapped = SECTION_TAG_MAP.get(tag_raw, None)
            # Check if it's a natural control line (e.g. "[Intro - Instrumental]")
            if is_natural_control_line(stripped):
                # Natural control line → tag-with-content, keep its tokens
                segments.append(("control", stripped, current_section, len(stripped)))
                tag_line_count += 1
            elif mapped is not None:
                current_section = mapped
                segments.append(("tag", stripped, current_section, len(stripped)))
                tag_line_count += 1
            else:
                # Unknown bracketed line → treat as tag (keep tokens but no unit)
                segments.append(("tag", stripped, current_section, len(stripped)))
                tag_line_count += 1
        else:
            segments.append(("lyric", stripped, current_section, len(stripped)))
            lyric_line_count += 1

    if not any(seg[0] == "lyric" for seg in segments):
        # No real lyrics → fallback: everything is UNKNOWN silence
        fallback_unit = LyricUnit(
            unit_id=0, section="UNKNOWN", text=lyrics_text,
            char_count=len(lyrics_text.replace(" ", "")),
            token_indices=list(range(min(L, 100))),
            occurrence_id=0, is_silence=True, is_lyric=False,
        )
        debug = {
            "lyric_ratio": 0.0, "silence_ratio": 1.0,
            "tag_line_count": tag_line_count,
            "lyric_line_count": lyric_line_count,
            "total_segments": len(segments),
            "warning": "no_lyric_lines_found",
        }
        return [fallback_unit], np.full(L, -1.0, dtype=np.float32), debug

    # ── 2. Estimate token spans from character proportions ───────────────────
    total_chars = sum(s[3] for s in segments)

    # Cumulative ratio → token boundaries
    cum_ratio = 0.0
    boundaries = [0]
    for seg_type, content, section, cc in segments:
        cum_ratio += cc / max(total_chars, 1)
        bound = int(round(cum_ratio * L))
        boundaries.append(min(bound, L))
    boundaries[-1] = L  # ensure exact endpoint
    # Fix off-by-one: ensure monotonic
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + 1
    boundaries[-1] = L
    if boundaries[-1] > L:
        boundaries[-1] = L

    # ── 3. Build units ───────────────────────────────────────────────────────
    units: List[LyricUnit] = []
    next_unit_id = 0
    occurrence_counters: Dict[str, int] = {}
    token_pos = np.full(L, -1.0, dtype=np.float32)
    token_to_unit_arr = np.full(L, -1, dtype=np.int32)
    tag_control_mask = np.zeros(L, dtype=bool)  # True for tag tokens that are control (not in any unit)
    lyric_token_count = 0
    tag_token_count = 0
    section_tag_line_ids: List[int] = []  # unit_ids of tag-based silence units

    for i, (seg_type, content, section, cc) in enumerate(segments):
        start = boundaries[i]
        end = min(boundaries[i + 1], L)
        if end <= start:
            continue

        token_ids = list(range(int(start), int(end)))

        if seg_type == "tag":
            tag_token_count += len(token_ids)
            section_tag_line_ids.append(next_unit_id)

            if section in NON_LYRIC_SECTIONS:
                # Non-singing tag → silence unit, tag tokens are control tokens
                dw = ratios.get(section.lower(), 0.04) if section.lower() in ratios else 0.04
                units.append(LyricUnit(
                    unit_id=next_unit_id, section=section, text=content,
                    char_count=0, token_indices=token_ids,  # tag tokens → this silence unit
                    occurrence_id=0, is_silence=True, is_lyric=False,
                    duration_weight=dw,
                ))
                for t in token_ids:
                    if t < L:
                        token_to_unit_arr[t] = next_unit_id
                next_unit_id += 1
            else:
                # Lyric section tags (Verse, Chorus, etc.) → no unit created.
                # Their tokens are control tokens: token_to_unit stays -1,
                # tag_control_mask marks them as readable non-lyric tokens.
                for t in token_ids:
                    if t < L:
                        tag_control_mask[t] = True
            continue

        if seg_type == "control":
            # Natural control line → create a control unit with real tokens
            # Determine section from content
            content_lower = content.lower()
            ctrl_section = "UNKNOWN"
            if any(kw in content_lower for kw in ("intro",)):
                ctrl_section = "INTRO"
            elif any(kw in content_lower for kw in ("outro", "fadeout", "fade out")):
                ctrl_section = "OUTRO"
            elif any(kw in content_lower for kw in ("break", "interlude", "solo", "instrumental")):
                ctrl_section = "INSTRUMENTAL"

            dw = ratios.get(ctrl_section.lower(), 0.03) if ctrl_section.lower() in ratios else 0.03
            units.append(LyricUnit(
                unit_id=next_unit_id, section=ctrl_section, text=content,
                char_count=0, token_indices=token_ids,
                occurrence_id=0, is_silence=True, is_control=True, is_lyric=False,
                duration_weight=dw,
            ))
            for t in token_ids:
                if t < L:
                    token_to_unit_arr[t] = next_unit_id
            next_unit_id += 1
            continue

        if seg_type == "lyric":
            occ_id = occurrence_counters.get(section, 0)
            occurrence_counters[section] = occ_id + 1

            units.append(LyricUnit(
                unit_id=next_unit_id, section=section, text=content,
                char_count=len(content.replace(" ", "")),
                token_indices=token_ids,
                occurrence_id=occ_id, is_silence=False, is_lyric=True,
            ))

            for t in token_ids:
                if t < L:
                    token_to_unit_arr[t] = next_unit_id
                    lyric_token_count += 1

            # Token position: simple linear within [0, 1]
            for j, t in enumerate(token_ids):
                if t < L:
                    token_pos[t] = j / max(len(token_ids) - 1, 1)

            next_unit_id += 1

    # ── 4. Auto-insert transition silence units ─────────────────────────────
    # Scan the unit sequence for section transitions and insert silence.
    # We work on the *current* unit list, inserting between adjacent lyric units.
    inserted_transitions: List[int] = []

    # Collect lyric-unit-only section sequence
    lyric_unit_seq = [(i, u) for i, u in enumerate(units) if not u.is_silence]

    # Add intro silence at beginning (skip if first unit is already intro)
    if len(lyric_unit_seq) > 0 and ratios.get("intro", 0) > 0:
        has_existing_intro = any(u.section == "INTRO" and u.is_silence for u in units[:3])
        if not has_existing_intro:
            intro_unit = LyricUnit(
                unit_id=next_unit_id, section="INTRO", text="",
                char_count=0, token_indices=[],
                occurrence_id=0, is_silence=True, is_lyric=False,
                duration_weight=ratios.get("intro", 0.06),
            )
            units.insert(0, intro_unit)
            next_unit_id += 1
            inserted_transitions.append(0)
            for u in units[1:]:
                u.unit_id += 1

    # Insert outro silence at end
    units.append(LyricUnit(
        unit_id=next_unit_id, section="OUTRO", text="",
        char_count=0, token_indices=[],
        occurrence_id=0, is_silence=True, is_lyric=False,
        duration_weight=ratios.get("outro", 0.06),
    ))
    next_unit_id += 1
    inserted_transitions.append(len(units) - 1)

    # Rebuild lyric_unit_seq after insertions
    lyric_unit_seq = [(i, u) for i, u in enumerate(units) if not u.is_silence]

    # Insert mid-song transitions
    auto_transition_defs = [
        ("CHORUS", "VERSE", ratios.get("chorus_to_verse", 0.04)),
        ("CHORUS", "BRIDGE", ratios.get("chorus_to_bridge", 0.03)),
        ("BRIDGE", "CHORUS", ratios.get("bridge_to_chorus", 0.03)),
    ]

    for idx in range(len(lyric_unit_seq) - 1):
        i_curr, u_curr = lyric_unit_seq[idx]
        i_next, u_next = lyric_unit_seq[idx + 1]
        # Check if this transition matches any auto-transition pattern
        for from_sec, to_sec, ratio in auto_transition_defs:
            if ratio <= 0:
                continue
            # Normalise section names for comparison
            curr_sec = u_curr.section.upper().replace("-", "").replace(" ", "")
            next_sec = u_next.section.upper().replace("-", "").replace(" ", "")
            from_norm = from_sec.upper().replace("-", "").replace(" ", "")
            to_norm = to_sec.upper().replace("-", "").replace(" ", "")
            if curr_sec == from_norm and next_sec == to_norm:
                # Insert silence unit between u_curr and u_next
                sil_unit = LyricUnit(
                    unit_id=next_unit_id,
                    section=f"TRANSITION_{from_sec}_{to_sec}",
                    text="", char_count=0, token_indices=[],
                    occurrence_id=0, is_silence=True, is_lyric=False,
                    duration_weight=ratio,
                )
                units.insert(i_next, sil_unit)
                next_unit_id += 1
                inserted_transitions.append(i_next)
                # Shift unit_ids for all units after insertion
                for j in range(i_next + 1, len(units)):
                    units[j].unit_id = j
                # Rebuild lyric_unit_seq
                lyric_unit_seq = [(i, u) for i, u in enumerate(units) if not u.is_silence]
                break

    # Re-assign unit_id cleanly
    for j, u in enumerate(units):
        u.unit_id = j

    # ── 5. Compute position for each token (scaffold positions) ──────────────
    valid_lyric_indices = np.where(token_to_unit_arr >= 0)[0]
    if len(valid_lyric_indices) > 1:
        for k, idx in enumerate(valid_lyric_indices):
            token_pos[idx] = k / (len(valid_lyric_indices) - 1)
    elif len(valid_lyric_indices) == 1:
        token_pos[valid_lyric_indices[0]] = 0.5

    # Return as torch Tensor for backward compat
    token_pos = torch.from_numpy(token_pos).float()

    # ── 6. Debug info ───────────────────────────────────────────────────────
    n_lyric_units = sum(1 for u in units if not u.is_silence)
    n_silence_units = sum(1 for u in units if u.is_silence)
    lyric_ratio = lyric_token_count / max(L, 1)
    silence_ratio = n_silence_units / max(len(units), 1)

    debug_info: Dict[str, Any] = {
        "lyric_ratio": lyric_ratio,
        "silence_ratio": silence_ratio,
        "total_units": len(units),
        "n_lyric_units": n_lyric_units,
        "n_silence_units": n_silence_units,
        "n_tag_silence_units": len([i for i in section_tag_line_ids if i < len(units) and units[i].is_silence]),
        "n_auto_transition_units": len(inserted_transitions),
        "tag_token_count": tag_token_count,
        "lyric_token_count": lyric_token_count,
        "tag_line_count": tag_line_count,
        "lyric_line_count": lyric_line_count,
        "section_occurrences": dict(occurrence_counters),
        "tag_control_mask": tag_control_mask,  # [L] bool for non-unit tag tokens
    }

    return units, token_pos, debug_info


def build_duration_scaffold(
    units: List[LyricUnit],
    text_len: int,
    gamma: float = 0.8,
    silence_multiplier: float = 0.5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    tag_control_mask: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """Build duration scaffold from a list of LyricUnits.

    Each lyric unit gets a weight proportional to ``char_count ** gamma``
    times a section-dependent multiplier.  Silence units get a fixed
    fraction of the average lyric weight.  All weights are normalised
    to sum to 1.

    Args:
        tag_control_mask: [L] bool array from ``parse_lyrics_to_units``.
            Tokens that are section tags (e.g. ``[Verse]``, ``[Chorus]``)
            but NOT assigned to any silence unit.  These are non-lyric but
            still readable by attention.

    Returns:
        dict with:
          - ``unit_boundaries`` [U + 1] normalised token boundaries
          - ``unit_duration`` [U] duration per unit
          - ``token_to_unit`` [L_text] unit id per text token (-1 = non-lyric)
          - ``lyric_mask`` [L_text] bool (only real lyric tokens)
          - ``control_mask`` [L_text] bool (tag/silence tokens readable by attention)
    """
    U = len(units)
    weights = torch.zeros(U, dtype=dtype)

    # Phase 1: auto-compute weights for non-explicit units
    explicit_indices = [i for i, u in enumerate(units) if u.duration_weight > 0]
    auto_indices = [i for i, u in enumerate(units) if u.duration_weight <= 0]
    lyric_auto = [i for i in auto_indices if not units[i].is_silence]

    total_auto_weight = 0.0
    if lyric_auto:
        for i in lyric_auto:
            u = units[i]
            mult = SECTION_MULTIPLIER.get(u.section.upper(), 1.0)
            w = max(u.char_count, 1) ** gamma * mult
            weights[i] = w
            total_auto_weight += w
        avg_lyric_weight = float(total_auto_weight) / len(lyric_auto)
        for i in auto_indices:
            u = units[i]
            if u.is_silence and i not in explicit_indices:
                weights[i] = avg_lyric_weight * silence_multiplier
                total_auto_weight += weights[i].item()
    elif not explicit_indices:
        weights[:] = 1.0 / U
        total_auto_weight = float(U)
    else:
        total_auto_weight = float(sum(weights[i].item() for i in auto_indices))

    # Phase 2: scale explicit-unit weights to hit their target duration ratios
    if explicit_indices:
        total_explicit_ratio = sum(units[i].duration_weight for i in explicit_indices)
        # Guard against over-allocation
        total_explicit_ratio = min(total_explicit_ratio, 0.85)
        for i in explicit_indices:
            r = units[i].duration_weight
            # weight = auto_weight * r / (1 - total_explicit_ratio)
            target_w = total_auto_weight * r / max(1.0 - total_explicit_ratio, 0.15)
            weights[i] = max(target_w, 1e-6)

    duration = weights / (weights.sum() + 1e-10)
    cumsum = torch.cumsum(duration, dim=0)
    unit_boundaries = torch.cat([torch.zeros(1, dtype=dtype), cumsum])  # [U + 1]
    unit_boundaries[-1] = 1.0  # numerical safety

    # token → unit mapping
    token_to_unit = torch.full((text_len,), -1, dtype=torch.long)
    for u in units:
        for idx in u.token_indices:
            if 0 <= idx < text_len:
                token_to_unit[idx] = u.unit_id

    # lyric_mask: only non-silence units
    silence_unit_ids = {u.unit_id for u in units if u.is_silence}
    lyric_mask = torch.zeros(text_len, dtype=torch.bool)
    for u in units:
        if not u.is_silence:
            for idx in u.token_indices:
                if 0 <= idx < text_len:
                    lyric_mask[idx] = True

    # control_mask: tag-based control tokens + tokens belonging to silence units
    control_mask = torch.zeros(text_len, dtype=torch.bool)
    if tag_control_mask is not None:
        for j in range(min(len(tag_control_mask), text_len)):
            if tag_control_mask[j]:
                control_mask[j] = True
    # Silence unit tokens are also control (but not lyric)
    for u in units:
        if u.is_silence:
            for idx in u.token_indices:
                if 0 <= idx < text_len:
                    control_mask[idx] = True

    # attendable_mask = lyric_mask | control_mask
    attendable_mask = lyric_mask | control_mask

    # lyric_unit_mask: True for units that are lyric (not silence)
    lyric_unit_mask = torch.zeros(U, dtype=torch.bool)
    # unit_section_ids: section ID per unit (for section embedding in retrieval)
    SECTION_ID_MAP_BUILD = {"UNKNOWN": 0, "INTRO": 1, "VERSE": 2, "PRECHORUS": 3,
                            "PRE-CHORUS": 3, "CHORUS": 4, "BRIDGE": 5,
                            "OUTRO": 6, "INSTRUMENTAL": 7}
    unit_section_ids = torch.zeros(U, dtype=torch.long)
    for i, u in enumerate(units):
        if not u.is_silence:
            lyric_unit_mask[i] = True
        unit_section_ids[i] = SECTION_ID_MAP_BUILD.get(u.section.upper(), 0)

    if device is not None:
        unit_boundaries = unit_boundaries.to(device)
        unit_duration = duration.to(device)
        token_to_unit = token_to_unit.to(device)
        lyric_mask = lyric_mask.to(device)
        control_mask = control_mask.to(device)
        attendable_mask = attendable_mask.to(device)
        lyric_unit_mask = lyric_unit_mask.to(device)
        unit_section_ids = unit_section_ids.to(device)

    return {
        "unit_boundaries": unit_boundaries,  # [U + 1]
        "unit_duration": duration,            # [U]
        "token_to_unit": token_to_unit,       # [L_text]
        "lyric_mask": lyric_mask,             # [L_text], only real lyric tokens
        "control_mask": control_mask,         # [L_text], tag/silence tokens readable by attention
        "attendable_mask": attendable_mask,   # [L_text], lyric_mask OR control_mask
        "lyric_unit_mask": lyric_unit_mask,   # [U] bool, True for non-silence units
        "unit_section_ids": unit_section_ids, # [U] long, section ID per unit
    }


# ===================================================================
#  Duration interval bias
# ===================================================================


def build_duration_interval_bias(
    p_final: torch.Tensor,
    unit_boundaries: torch.Tensor,
    token_to_unit: torch.Tensor,
    attendable_mask: torch.Tensor,
    sigma: float = 0.06,
    lambda_: float = 0.25,
    max_bias: float = 0.5,
) -> torch.Tensor:
    """Build a cross-attention bias that encourages each audio token to
    attend to text tokens whose *duration interval* contains the current
    progress position ``p_final[i]``.

    Bias is applied to ALL attendable tokens (lyric + control) based on
    the duration interval of the unit they belong to.
    Non-attendable tokens (no unit) get zero bias.

    Args:
        p_final: [B, T] monotonic progress in [0, 1].
        unit_boundaries: [U + 1] normalised boundaries.
        token_to_unit: [L] or [B, L] unit id per text token (-1 = non-lyric).
        attendable_mask: [L] or [B, L] bool, True for both lyric and control tokens.
        sigma: Gaussian width (logit-space).
        lambda_: Bias strength multiplier.
        max_bias: Maximum absolute clamp value.

    Returns:
        bias: [B, T, L] additive logit bias (non-positive for attendable tokens,
              zero for non-attendable tokens).
    """
    device = p_final.device
    dtype = p_final.dtype
    B, T = p_final.shape

    # Normalise inputs
    if token_to_unit.dim() == 1:
        token_to_unit = token_to_unit.unsqueeze(0).expand(B, -1)
    if attendable_mask.dim() == 1:
        attendable_mask = attendable_mask.unsqueeze(0).expand(B, -1)

    L = token_to_unit.shape[-1]
    U = unit_boundaries.shape[-1] - 1

    # Boundaries: left_u = unit_boundaries[u], right_u = unit_boundaries[u+1]
    left = unit_boundaries[:-1]  # [U]
    right = unit_boundaries[1:]  # [U]
    centre = (left + right) / 2  # [U]

    # Distance from p_final[i] to unit centre
    p_exp = p_final.unsqueeze(-1)  # [B, T, 1]
    centre_exp = centre.unsqueeze(0).unsqueeze(0)  # [1, 1, U]
    dist_centre = (p_exp - centre_exp).abs()  # [B, T, U]

    # Per-unit bias
    unit_bias = -lambda_ * (dist_centre / sigma) ** 2
    unit_bias = unit_bias.clamp(min=-max_bias, max=0.0)

    # Gather unit_bias by token_to_unit (applies to both lyric and control tokens)
    unit_ids = token_to_unit.unsqueeze(1).expand(-1, T, -1)
    unit_ids = unit_ids.clamp(min=0, max=U - 1)
    bias = torch.gather(unit_bias, dim=-1, index=unit_ids)

    # Zero out for non-attendable tokens
    attendable_exp = attendable_mask.unsqueeze(1).expand(-1, T, -1).float()
    bias = bias * attendable_exp.float()

    return bias


# ===================================================================
#  Mass-preserving attention (split-softmax)
# ===================================================================


def mass_preserving_attention(
    logits: torch.Tensor,
    value: torch.Tensor,
    lyric_mask: torch.Tensor,
    bias: torch.Tensor,
    gate: float = 0.1,
    control_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Text-mass-preserving split-softmax attention.

    Preserves total mass on *text tokens* (lyric + control), not just lyric.
    This allows lyric mass to drop during silence intervals while control
    tokens (section tags) naturally absorb the freed mass.

    Without ``control_mask``, falls back to lyric-mass-preserving (old behaviour).

    Args:
        logits: [B, H, T, L] pre-softmax attention scores.
        value: [B, H, L, Dh] value states.
        lyric_mask: [L] or [B, L] bool, True for lyric tokens.
        bias: [B, T, L] additive bias (non-positive for lyric).
        gate: Bias strength multiplier (0 = off).
        control_mask: [L] or [B, L] bool, True for tag/silence tokens.
            These are non-lyric but are valid attention targets.

    Returns:
        output: [B, H, T, Dh] attended values.
        attn_new: [B, H, T, L] new attention weights (sum to 1).
        stats: dict with text_mass_base, lyric_mass_base, lyric_mass_new, etc.
    """
    B, H, T, L = logits.shape
    device = logits.device

    # Expand masks to [B, H, T, L]
    def _expand(m):
        if m.dim() == 1:
            return m.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, H, T, -1)
        return m.unsqueeze(1).unsqueeze(1).expand(B, H, T, -1)

    lyric_exp = _expand(lyric_mask).bool()
    lyric_float = lyric_exp.float()

    if control_mask is not None:
        ctrl_exp = _expand(control_mask).bool()
        text_exp = lyric_exp | ctrl_exp
    else:
        ctrl_exp = None
        text_exp = lyric_exp

    text_float = text_exp.float()

    # Baseline: base attention mass on text tokens
    attn_base = F.softmax(logits, dim=-1, dtype=torch.float32)
    text_mass_base = (attn_base * text_float).sum(dim=-1)  # [B, H, T]
    lyric_mass_base = (attn_base * lyric_float).sum(dim=-1)

    # Apply bias (only affects lyric logits)
    bias_4d = bias.unsqueeze(1)  # [B, 1, T, L]
    biased_logits = logits + gate * bias_4d

    # Text split-softmax: lyric tokens get biased, control tokens get original
    if ctrl_exp is not None:
        # Lyric: use biased logits, control: use original logits, combined softmax
        text_logits = torch.where(lyric_exp, biased_logits, logits)
        text_logits = text_logits.masked_fill(~text_exp, float("-inf"))
    else:
        text_logits = biased_logits.masked_fill(~text_exp, float("-inf"))

    attn_text = F.softmax(text_logits, dim=-1, dtype=torch.float32)
    attn_text = attn_text.masked_fill(~text_exp, 0.0)

    # Non-text split-softmax
    non_text_logits = logits.masked_fill(text_exp, float("-inf"))
    attn_non_text = F.softmax(non_text_logits, dim=-1, dtype=torch.float32)
    attn_non_text = attn_non_text.masked_fill(text_exp, 0.0)

    # Recombine preserving total text mass
    attn_new = attn_text * text_mass_base.unsqueeze(-1) + \
               attn_non_text * (1.0 - text_mass_base).unsqueeze(-1)
    attn_new = attn_new / (attn_new.sum(dim=-1, keepdim=True) + 1e-10)

    lyric_mass_new = (attn_new * lyric_float).sum(dim=-1)
    text_mass_new = (attn_new * text_float).sum(dim=-1)

    output = torch.matmul(attn_new.to(value.dtype), value)

    stats = {
        "text_mass_base": text_mass_base[0, 0].mean().item(),
        "text_mass_new": text_mass_new[0, 0].mean().item(),
        "text_mass_delta": (text_mass_new - text_mass_base).abs().mean().item(),
        "lyric_mass_base": lyric_mass_base[0, 0].mean().item(),
        "lyric_mass_new": lyric_mass_new[0, 0].mean().item(),
        "lyric_mass_delta": (lyric_mass_new - lyric_mass_base).abs().mean().item(),
    }

    return output, attn_new, stats


# ===================================================================
#  PMDC regularisation loss (optional)
# ===================================================================


def pmdc_regularization_loss(
    p_final: torch.Tensor,
    p_base: torch.Tensor,
    s_pm: torch.Tensor,
    lambda_clock: float = 0.02,
    lambda_res: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Optional regularisation loss for PMDC training.

    ``L_clock`` penalises deviation from the base progress schedule.
    ``L_res`` penalises large log-speed residuals (keeps PM output small).

    Returns:
        loss: scalar regularisation loss.
        stats: dict with ``L_clock``, ``L_res``.
    """
    L_clock = F.smooth_l1_loss(p_final, p_base)
    L_res = s_pm.pow(2).mean()
    loss = lambda_clock * L_clock + lambda_res * L_res
    return loss, {"L_clock": L_clock.item(), "L_res": L_res.item()}


# ===================================================================
#  Freeze helpers
# ===================================================================


def freeze_except_pmdc(model: nn.Module) -> Tuple[int, int]:
    """Freeze all parameters except ``PhaseMemoryDurationClock`` modules.

    Args:
        model: Model containing PhaseMemoryDurationClock sub-modules.

    Returns:
        (trainable_params, total_params).
    """
    trainable_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, PhaseMemoryDurationClock):
            for param in module.parameters():
                trainable_ids.add(id(param))

    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if id(param) in trainable_ids:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False

    return trainable, total


def freeze_except_phase_memory(model: nn.Module) -> Tuple[int, int]:
    """Alias for ``freeze_except_pmdc`` (old name, kept for compatibility)."""
    return freeze_except_pmdc(model)


def unfreeze_all(model: nn.Module) -> None:
    """Enable gradient computation for all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def reset_phase_memory(model: nn.Module) -> None:
    """Reset all PhaseMemoryDurationClock modules inside a model.

    Note: PMDC has no persistent state, so this is a no-op.
    Included for backward compatibility.
    """
    pass


def set_phase_memory_scale(model: nn.Module, scale: float) -> None:
    """Set beta scaling factor on all PMDC modules.

    Legacy compatibility: old PhaseMemory had a ``phase_scale`` attribute.
    PMDC uses ``beta`` instead.
    """
    for module in model.modules():
        if isinstance(module, PhaseMemoryDurationClock):
            module.beta = scale


# ===================================================================
#  PMDCResidualClock freeze helper
# ===================================================================

def freeze_except_pmdc_clock(
    model: nn.Module,
    pmdc_clock: Optional[nn.Module] = None,
) -> Tuple[int, int, List[str]]:
    """Freeze all model params except ``PMDCResidualClock`` sub-modules.

    Args:
        model: DiT model (all params frozen).
        pmdc_clock: Optional standalone ``PMDCResidualClock`` module that
            is not embedded in the model.  Its params are set to trainable.

    Returns:
        (trainable_params, total_params, trainable_names).
    """
    trainable_ids: set[int] = set()
    trainable_names: List[str] = []

    # Check model for embedded PMDCResidualClock modules
    for name, module in model.named_modules():
        if isinstance(module, PMDCResidualClock):
            for param in module.parameters():
                trainable_ids.add(id(param))
                trainable_names.append(f"model.{name}.{id(param)}")

    # Check standalone PMDCResidualClock
    if pmdc_clock is not None:
        for param in pmdc_clock.parameters():
            trainable_ids.add(id(param))
            trainable_names.append(f"pmdc_clock.<params>")

    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if id(param) in trainable_ids:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    if pmdc_clock is not None:
        for param in pmdc_clock.parameters():
            total += param.numel()
            if id(param) in trainable_ids:
                trainable += param.numel()

    return trainable, total, trainable_names


# ===================================================================
#  Smoke test
# ===================================================================


def smoke_test_pmdc():
    """Run minimal verification of all PMDC components."""
    print("=" * 60)
    print("PMDC smoke test")
    print("=" * 60)

    B, T, D = 2, 128, 2048
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. PhaseMemoryDurationClock forward --------------------------------
    print("\n[1] PhaseMemoryDurationClock forward...")
    h = torch.randn(B, T, D, device=device)
    p_base = torch.linspace(0, 1, T, device=device).unsqueeze(0).expand(B, T)

    clock = PhaseMemoryDurationClock(D, mem_dim=128, hidden_dim=256, beta=0.1).to(device)
    s_pm, p_final = clock(h, p_base)

    assert s_pm.shape == (B, T), f"s_pm shape: {s_pm.shape}"
    assert p_final.shape == (B, T), f"p_final shape: {p_final.shape}"
    assert torch.all(p_final[:, 1:] >= p_final[:, :-1] - 1e-6), "p_final not monotonic"
    assert p_final.min() >= -1e-5, f"p_final min: {p_final.min()}"
    assert p_final.max() <= 1.00001, f"p_final max: {p_final.max()}"

    mae = (p_final - p_base).abs().mean().item()
    print(f"  s_pm range: [{s_pm.min().item():.4f}, {s_pm.max().item():.4f}]")
    print(f"  p_final MAE vs p_base: {mae:.6f}")
    if mae < 0.05:
        print("  ✓ p_final ≈ p_base at init (zero-init working)")
    else:
        print("  ⚠ p_final deviates from p_base (may be OK if beta > 0)")

    # ---- 2. PhaseMemoryDurationClock without p_base -------------------------
    print("\n[2] PhaseMemoryDurationClock without p_base...")
    s_pm2, p_final2 = clock(h)
    assert p_final2.shape == (B, T)
    assert torch.all(p_final2[:, 1:] >= p_final2[:, :-1] - 1e-6)
    print("  ✓ default linear p_base works")

    # ---- 3. parse_lyrics_to_units — tag-aware parsing ------------------------
    print("\n[3] Tag-aware lyrics parsing...")

    # Test with structured lyrics containing section tags
    test_lyrics = """[Intro]

[Verse]
hello world
this is verse one

[Pre-chorus]
building up

[Chorus]
sing the chorus
la la la

[Verse]
verse two here

[Chorus]
final chorus

[Outro]"""

    # Create a realistic section_ids array (simulating LyricsStructureParser output)
    # We use 128 tokens as a proxy for L_eff
    L_test = 128
    # Simulate: first few tokens are UNKNOWN (0), then INTRO(1), VERSE(2), PRECHORUS(3), CHORUS(4), etc.
    fake_ids = torch.zeros(L_test, dtype=torch.long)
    fake_ids[:10] = 0          # UNKNOWN prefix
    fake_ids[10:30] = 2        # VERSE
    fake_ids[30:45] = 3        # PRECHORUS
    fake_ids[45:70] = 4        # CHORUS
    fake_ids[70:90] = 2        # VERSE
    fake_ids[90:120] = 4       # CHORUS
    fake_ids[120:] = 6         # OUTRO

    units_new, pos_new, debug_new = parse_lyrics_to_units(test_lyrics, fake_ids)

    # Verify lyric section tags (Verse, Chorus, Bridge, Pre-chorus) are NOT units.
    # Non-singing tags (Intro, Outro) correctly create silence units.
    lyric_tag_units = [u for u in units_new
                       if u.text.strip().startswith("[")
                       and not u.is_silence]
    assert len(lyric_tag_units) == 0, \
        f"lyric section tags found as non-silence units: {lyric_tag_units}"

    # Verify silence units exist
    n_silence = sum(1 for u in units_new if u.is_silence)

    # Verify lyric units only contain real lyrics
    for u in units_new:
        if not u.is_silence:
            assert not u.text.strip().startswith("["), \
                f"lyric unit contains tag: {u.text}"
            assert len(u.token_indices) > 0, \
                f"lyric unit has no tokens: {u.text}"

    # Verify section assignment is correct
    # First lyric should be VERSE (after [Verse])
    first_lyric = [u for u in units_new if not u.is_silence]
    assert len(first_lyric) > 0, "no lyric units"
    assert first_lyric[0].section == "VERSE", \
        f"first lyric section {first_lyric[0].section} != VERSE"
    # Prechorus lyric should be PRECHORUS
    prechorus_units = [u for u in units_new if u.section == "PRECHORUS"]
    assert len(prechorus_units) >= 1, "no PRECHORUS units"

    # Verify occurrence counting
    chorus_units = [u for u in units_new if u.section == "CHORUS"]
    occ_ids = [u.occurrence_id for u in chorus_units if not u.is_silence]
    assert len(set(occ_ids)) == len(occ_ids), \
        f"duplicate occurrence ids: {occ_ids}"
    assert len(occ_ids) >= 2, \
        f"expected 2+ CHORUS occurrences, got {len(occ_ids)}"

    print(f"  ✓ units={len(units_new)}, lyric={debug_new['n_lyric_units']}, "
          f"silence={n_silence}, transitions={debug_new['n_auto_transition_units']}")
    print(f"  ✓ section_occurrences: {debug_new['section_occurrences']}")

    # ---- 4. Duration scaffold from new units --------------------------------
    print("\n[4] Duration scaffold from tag-aware units...")
    # Get tag_control_mask from debug_info
    tcm = debug_new.get("tag_control_mask", None)
    scaffold = build_duration_scaffold(units_new, text_len=L_test, device=device,
                                        tag_control_mask=tcm)
    boundaries = scaffold["unit_boundaries"]
    duration = scaffold["unit_duration"]
    t2u = scaffold["token_to_unit"]
    lyric_mask = scaffold["lyric_mask"]
    control_mask = scaffold["control_mask"]

    assert abs(boundaries[0].item()) < 1e-5, "boundaries[0] != 0"
    assert abs(boundaries[-1].item() - 1.0) < 1e-5, f"boundaries[-1] {boundaries[-1].item()} != 1"
    assert abs(duration.sum().item() - 1.0) < 1e-5, "duration.sum() != 1"

    # Verify lyric_mask excludes all tag/silence tokens
    for u in units_new:
        if u.is_silence and any(0 <= idx < L_test for idx in u.token_indices):
            for idx in u.token_indices:
                if 0 <= idx < L_test:
                    assert not lyric_mask[idx], f"silence token {idx} in lyric_mask"

    # Verify control_mask includes tag control tokens
    n_ctrl = control_mask.sum().item()
    assert n_ctrl > 0, "no control tokens"

    # Verify lyric_mask and control_mask are disjoint
    assert (lyric_mask & control_mask).sum().item() == 0, \
        "lyric_mask and control_mask overlap"

    # Total silence duration should be reasonable
    silence_duration = sum(duration[i].item() for i, u in enumerate(units_new) if u.is_silence)
    assert 0.05 <= silence_duration <= 0.40, \
        f"silence_duration {silence_duration:.3f} not in [0.05, 0.40]"

    print(f"  ✓ boundaries shape: {boundaries.shape}, sum(duration)={duration.sum().item():.6f}")
    print(f"  ✓ silence_duration={silence_duration:.3f} ({silence_duration*100:.1f}%)")
    print(f"  ✓ lyric_mask (true)={lyric_mask.sum().item()}, control_mask (true)={n_ctrl}")

    # ---- 5. Duration interval bias ------------------------------------------
    print("\n[5] Duration interval bias (with attendable_mask)...")
    bias = build_duration_interval_bias(
        p_final=p_final,
        unit_boundaries=boundaries,
        token_to_unit=t2u,
        attendable_mask=scaffold["attendable_mask"],
    )
    assert bias.shape[0] == B and bias.shape[1] == T, f"bias shape batch/time: {bias.shape}"
    assert bias.shape[2] == L_test, f"bias shape L: {bias.shape}"
    assert (bias <= 1e-6).all(), "bias should be non-positive"
    # Both lyric AND control tokens are attendable → should have bias
    attendable = scaffold["attendable_mask"]
    assert (bias[:, :, attendable] <= 0).all(), "attendable tokens should have bias ≤ 0"
    non_attendable = ~attendable
    assert (bias[:, :, non_attendable] == 0).all(), "non-attendable tokens should have 0 bias"
    print(f"  ✓ bias shape: {bias.shape}, range: [{bias.min():.4f}, {bias.max():.4f}]")

    # ---- 6. Text-mass-preserving attention ----------------------------------
    print("\n[6] Text-mass-preserving attention...")
    n_heads = 4
    head_dim = 64
    logits = torch.randn(B, n_heads, T, L_test, device=device)
    value = torch.randn(B, n_heads, L_test, head_dim, device=device)

    out, attn_new, stats = mass_preserving_attention(
        logits=logits,
        value=value,
        lyric_mask=lyric_mask,
        control_mask=control_mask,
        bias=bias,
        gate=0.1,
    )

    assert out.shape == (B, n_heads, T, head_dim), f"out shape: {out.shape}"
    assert attn_new.shape == (B, n_heads, T, L_test), f"attn shape: {attn_new.shape}"
    attn_sum = attn_new.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4), \
        f"attn sum not 1: max dev {((attn_sum - 1).abs().max().item())}"
    print(f"  ✓ out shape: {out.shape}")
    print(f"  ✓ attn sums to 1: max dev {((attn_new.sum(-1) - 1).abs().max().item()):.6f}")
    print(f"  ✓ text_mass_delta: {stats.get('text_mass_delta', 0):.6f}")
    print(f"  ✓ lyric_mass_delta: {stats['lyric_mass_delta']:.6f}")

    # ---- 7. Regularisation loss ---------------------------------------------
    print("\n[7] Regularisation loss...")
    loss, loss_stats = pmdc_regularization_loss(p_final, p_base, s_pm)
    assert torch.isfinite(loss), "loss is not finite"
    print(f"  ✓ L_total={loss.item():.6f}, L_clock={loss_stats['L_clock']:.6f}, L_res={loss_stats['L_res']:.6f}")

    # ---- 8. Freeze helper ---------------------------------------------------
    print("\n[8] Freeze helper...")
    # Simulate a model with PMDC as submodule
    dummy_model = nn.ModuleList([nn.Linear(10, 10), clock])
    trainable, total = freeze_except_pmdc(dummy_model)
    print(f"  ✓ trainable={trainable}, total={total}")
    assert trainable > 0, "no trainable params in PMDC"
    assert trainable < total, "non-PMDC params should be frozen"

    print("\n" + "=" * 60)
    print("ALL 8/8 smoke tests passed ✓")
    print("=" * 60)


# ===================================================================
#  PMRetrievalPhaseMemory — hidden-residual PM (new, standalone)
# ===================================================================


class PMRetrievalPhaseMemory(nn.Module):
    """
    PhaseMemory state extractor for PM-conditioned lyric retrieval.

    It reads layer hidden states and returns a differentiable pm_state.
    It must not inject hidden residual by default.
    """

    def __init__(
        self,
        dim: int,
        mem_dim: int = 128,
        hidden_dim: int = 256,
        normalize_internal_state: bool = True,
    ):
        super().__init__()
        self.mem_dim = mem_dim
        self.normalize_internal_state = normalize_internal_state

        self.input_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
        )
        self.proj_r = nn.Linear(hidden_dim, mem_dim)
        self.proj_i = nn.Linear(hidden_dim, mem_dim)
        self.omega = nn.Linear(hidden_dim + 2 * mem_dim, mem_dim)

    @property
    def pm_dim(self) -> int:
        return 2 * self.mem_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        diffusion_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D]

        Returns:
            pm_state: [B, T, 2 * mem_dim]
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        h_pool = hidden_states.mean(dim=1)  # [B, D]
        x = self.input_proj(h_pool)         # [B, hidden_dim]

        zr = torch.zeros(B, self.mem_dim, device=device, dtype=dtype)
        zi = torch.zeros(B, self.mem_dim, device=device, dtype=dtype)

        inp_r = self.proj_r(x)
        inp_i = self.proj_i(x)

        omega_in = torch.cat([x, zr, zi], dim=-1)
        omega = math.pi * torch.tanh(self.omega(omega_in))

        c = torch.cos(omega)
        s = torch.sin(omega)

        # IMPORTANT: use old zr/zi for both updates
        zr_old, zi_old = zr, zi
        zr_new = zr_old * c - zi_old * s + inp_r
        zi_new = zr_old * s + zi_old * c + inp_i

        if self.normalize_internal_state:
            scale = torch.sqrt(zr_new ** 2 + zi_new ** 2 + 1.0)
            zr_new = zr_new / scale
            zi_new = zi_new / scale

        pm_state = torch.cat([zr_new, zi_new], dim=-1)  # [B, 2*mem_dim]
        pm_state = pm_state.unsqueeze(1).expand(-1, T, -1)

        # Do NOT detach. Gradients must flow from adapter to PM.
        return pm_state


# ===================================================================
#  LyricRetrievalAdapter — auxiliary lyric retrieval via side branch
# ===================================================================


def _corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation coefficient between two flattened tensors."""
    xf = x.flatten().float()
    yf = y.flatten().float()
    xm = xf.mean()
    ym = yf.mean()
    xd = xf - xm
    yd = yf - ym
    num = (xd * yd).sum()
    den = torch.sqrt((xd ** 2).sum() * (yd ** 2).sum() + 1e-8)
    return num / den


class PhaseScaffoldWarp(nn.Module):
    """PM-conditioned phase warp for dynamic scaffold bias.

    Predicts a small, smooth, low-frequency offset to p_audio based on
    PM state and timestep embedding, then builds a phase bias for
    retrieval attention.
    """

    def __init__(
        self,
        pm_dim: int = 256,
        time_dim: int = 128,
        hidden_dim: int = 128,
        phase_num_freqs: int = 4,
        phase_offset_max: float = 0.05,
        phase_bias_sigma: float = 0.18,
        phase_bias_clamp_min: float = -2.0,
        phase_bias_dropout: float = 0.3,
    ):
        super().__init__()
        self.phase_num_freqs = phase_num_freqs
        self.phase_offset_max = phase_offset_max
        self.phase_bias_sigma = phase_bias_sigma
        self.phase_bias_clamp_min = phase_bias_clamp_min
        self.phase_bias_dropout = phase_bias_dropout

        # Control MLP: PM_global + timestep_global → Fourier coefficients
        ctrl_dim = pm_dim + time_dim
        self.ctrl_norm = nn.LayerNorm(ctrl_dim)
        self.coef_head = nn.Sequential(
            nn.Linear(ctrl_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * phase_num_freqs),
        )

        # Zero init so initial coef=0 → delta_p=0 → p_phase=p_audio
        nn.init.zeros_(self.coef_head[-1].weight)
        nn.init.zeros_(self.coef_head[-1].bias)

        # Register constant freqs buffer [1, 2, ..., M]
        self.register_buffer("freqs", torch.arange(1, phase_num_freqs + 1, dtype=torch.float32))

    def forward(
        self,
        pm_state: torch.Tensor,
        p_audio: torch.Tensor,
        c_text: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pm_state: [B, T, pm_dim] PM phase state.
            p_audio: [B, T] audio progress in [0, 1].
            c_text: [B, L] text token positions in [0, 1].
            timestep_emb: [B, T, time_dim] or [B, time_dim].

        Returns:
            phase_bias: [B, T, L] non-positive bias for retrieval score.
            p_phase: [B, T] warped progress.
            delta_p: [B, T] predicted offset.
            diag: dict of diagnostics.
        """
        B, T = p_audio.shape
        device = p_audio.device
        dtype = p_audio.dtype

        # ---- Global control vector -------------------------------------------
        pm_global = pm_state.mean(dim=1)  # [B, pm_dim]

        if timestep_emb is None:
            t_global = torch.zeros(B, 128, device=device, dtype=dtype)
        elif timestep_emb.dim() == 3:
            t_global = timestep_emb.mean(dim=1)
        else:
            t_global = timestep_emb

        ctrl = torch.cat([pm_global, t_global], dim=-1)
        ctrl = self.ctrl_norm(ctrl)

        # ---- Fourier coefficients ---------------------------------------------
        coef = self.coef_head(ctrl)  # [B, 2*M]
        sin_coef, cos_coef = coef.chunk(2, dim=-1)  # each [B, M]

        # ---- Construct basis: sin(2π*m*p) and cos(2π*m*p) --------------------
        # p_audio: [B, T, 1], freqs: [M] → phase: [B, T, M]
        phase = 2 * math.pi * p_audio.unsqueeze(-1) * self.freqs  # [B, T, M]

        sin_basis = torch.sin(phase)    # [B, T, M]
        cos_basis = torch.cos(phase)    # [B, T, M]

        # Expand coefs: [B, 1, M] * [B, T, M] → sum → [B, T]
        raw_delta = (
            (sin_coef.unsqueeze(1) * sin_basis).sum(dim=-1)
            + (cos_coef.unsqueeze(1) * cos_basis).sum(dim=-1)
        )

        # ---- Bound and apply --------------------------------------------------
        delta_p = self.phase_offset_max * torch.tanh(raw_delta)
        p_phase = torch.clamp(p_audio + delta_p, 0.0, 1.0)

        # ---- Phase bias -------------------------------------------------------
        # [B, T, 1] - [B, 1, L] → [B, T, L]
        dist = p_phase[:, :, None] - c_text[:, None, :]
        phase_bias = -(dist / self.phase_bias_sigma) ** 2
        phase_bias = torch.clamp(phase_bias, min=self.phase_bias_clamp_min, max=0.0)

        # ---- Dropout -----------------------------------------------------------
        bias_enabled = True
        if self.training and self.phase_bias_dropout > 0:
            if torch.rand((), device=phase_bias.device) < self.phase_bias_dropout:
                bias_enabled = False
                phase_bias = torch.zeros_like(phase_bias)

        # ---- Diagnostics -------------------------------------------------------
        with torch.no_grad():
            diag = {
                "delta_p_mean": delta_p.mean().item(),
                "delta_p_abs_mean": delta_p.abs().mean().item(),
                "delta_p_std": delta_p.std().item(),
                "delta_p_max_abs": delta_p.abs().max().item(),
                "p_phase_min": p_phase.min().item(),
                "p_phase_max": p_phase.max().item(),
                "phase_bias_std": phase_bias.std().item(),
                "phase_bias_enabled": float(bias_enabled),
                "coef_norm": coef.norm().item(),
            }

        return phase_bias, p_phase, delta_p, diag


class LyricRetrievalAdapter(nn.Module):
    """PM-conditioned lyric retrieval adapter.

    A side branch that reads PM state + lyric scaffold coordinates
    and produces a gated hidden residual.  The original cross-attention
    is untouched.

    Key design choices:
      - pm_state is LayerNorm'd before q_mlp.
      - q/k are L2-normalised before dot product (cosine similarity).
      - A learnable ``qk_score_scale`` replaces fixed 1/sqrt(d_r).
      - Weak scaffold prior helps initial alignment.
      - out_proj is tiny-normal initialised (not zero), so gradient flows
        from step 1.
    """

    def __init__(
        self,
        hidden_dim: int,
        text_dim: int,
        pm_dim: int,
        d_r: int = 64,
        coord_dim: int = 64,
        time_dim: int = 128,
        num_sections: int = 8,
        num_token_types: int = 4,

        # ---- v3 (deprecated in v4, kept for compat) --------------------------
        residual_scale: float = 0.1,
        gamma_init: float = 0.1,

        # ---- v3 prior config (kept) ------------------------------------------
        qk_score_scale_init: float = 2.0,
        use_adapter_scaffold_prior: bool = True,
        adapter_prior_sigma: float = 0.18,
        adapter_prior_lambda: float = 0.2,
        adapter_prior_clamp_min: float = -2.0,
        adapter_prior_dropout: float = 0.3,

        # ---- v4: Phase bias config -------------------------------------------
        use_phase_bias: bool = True,
        phase_bias_lambda: float = 0.03,
        phase_offset_max: float = 0.05,
        phase_num_freqs: int = 4,
        phase_bias_sigma: float = 0.18,
        phase_bias_clamp_min: float = -2.0,
        phase_bias_dropout: float = 0.3,

        # ---- v4: RMS writer config -------------------------------------------
        use_rms_writer: bool = True,
        write_alpha_init: float = 1e-4,
        write_alpha_max: float = 1e-3,
        writer_eps: float = 1e-6,
    ):
        super().__init__()
        self.d_r = d_r
        self.time_dim = time_dim

        # v3 prior config
        self.residual_scale = residual_scale
        self.use_adapter_scaffold_prior = use_adapter_scaffold_prior
        self.adapter_prior_sigma = adapter_prior_sigma
        self.adapter_prior_lambda = adapter_prior_lambda
        self.adapter_prior_clamp_min = adapter_prior_clamp_min
        self.adapter_prior_dropout = adapter_prior_dropout

        # v4 config
        self.use_phase_bias = use_phase_bias
        self.phase_bias_lambda = phase_bias_lambda
        self.use_rms_writer = use_rms_writer
        self.write_alpha_max = write_alpha_max
        self.writer_eps = writer_eps

        # Warn if both phase_bias and scaffold_prior are on
        if use_phase_bias and use_adapter_scaffold_prior:
            import warnings as _w
            _w.warn(
                "use_phase_bias=True and use_adapter_scaffold_prior=True are mutually exclusive. "
                "Disabling scaffold_prior in favor of phase_bias."
            )
            self.use_adapter_scaffold_prior = False
            self.use_phase_bias = True

        # PM state LayerNorm
        self.pm_state_norm = nn.LayerNorm(pm_dim)

        # Audio-side coordinate MLP
        self.audio_coord_mlp = nn.Sequential(
            nn.Linear(3, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
            nn.SiLU(),
        )

        # Text-side coordinate MLP
        self.text_coord_mlp = nn.Sequential(
            nn.Linear(3, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
            nn.SiLU(),
        )

        # Text-side auxiliary embeddings
        self.section_embedding = nn.Embedding(num_sections, coord_dim // 2)
        self.token_type_embedding = nn.Embedding(num_token_types, coord_dim // 2)

        # Query MLP (audio → retrieval query)
        q_in_dim = pm_dim + coord_dim + time_dim
        self.q_mlp = nn.Sequential(
            nn.Linear(q_in_dim, d_r * 2),
            nn.SiLU(),
            nn.Linear(d_r * 2, d_r),
        )

        # Key MLP (text → retrieval key)
        k_in_dim = text_dim + coord_dim + coord_dim // 2 + coord_dim // 2
        self.k_mlp = nn.Sequential(
            nn.Linear(k_in_dim, d_r * 2),
            nn.SiLU(),
            nn.Linear(d_r * 2, d_r),
        )

        # Value MLP (text → retrieval value)
        self.v_mlp = nn.Sequential(
            nn.Linear(text_dim, d_r * 2),
            nn.SiLU(),
            nn.Linear(d_r * 2, d_r),
        )

        # Learnable score scale
        self.logit_score_scale = nn.Parameter(torch.tensor(math.log(max(qk_score_scale_init, 1e-6))))

        # Output projection
        self.out_proj = nn.Linear(d_r, hidden_dim)
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

        # v3 gate (kept but not used by default in v4)
        self.gamma_r = nn.Parameter(torch.tensor(float(gamma_init)))

        # --- v4: PhaseScaffoldWarp --------------------------------------------
        self.phase_scaffold_warp = PhaseScaffoldWarp(
            pm_dim=pm_dim,
            time_dim=time_dim,
            hidden_dim=128,
            phase_num_freqs=phase_num_freqs,
            phase_offset_max=phase_offset_max,
            phase_bias_sigma=phase_bias_sigma,
            phase_bias_clamp_min=phase_bias_clamp_min,
            phase_bias_dropout=phase_bias_dropout,
        )

        # --- v4: RMS writer ---------------------------------------------------
        init_prob = max(write_alpha_init / max(write_alpha_max, 1e-10), 1e-6)
        init_prob = min(init_prob, 0.999)
        self.write_logit = nn.Parameter(torch.tensor(math.log(init_prob / (1.0 - init_prob))))

    @property
    def qk_score_scale(self) -> torch.Tensor:
        return torch.exp(self.logit_score_scale)

    @property
    def write_alpha(self) -> torch.Tensor:
        return self.write_alpha_max * torch.sigmoid(self.write_logit)

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_hidden: torch.Tensor,
        pm_state: torch.Tensor,
        p_audio: torch.Tensor,
        c_text: torch.Tensor,
        section_id: torch.Tensor,
        token_type_id: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_scaffold_prior: Optional[bool] = None,
        use_phase_bias: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B = hidden_states.shape[0]
        T_a = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # ---- PM state LayerNorm with diagnostic ----------------------------
        pm_state_raw = pm_state
        pm_state = self.pm_state_norm(pm_state)
        pm_state_norm_before = pm_state_raw.norm(dim=-1).mean()
        pm_state_norm_after = pm_state.norm(dim=-1).mean()

        # ---- Audio-side coordinate embedding -------------------------------
        a_feat = torch.stack([p_audio, p_audio ** 2, 1.0 - p_audio], dim=-1)
        audio_coord = self.audio_coord_mlp(a_feat)

        # ---- Text-side coordinate embedding --------------------------------
        t_feat = torch.stack([c_text, c_text ** 2, 1.0 - c_text], dim=-1)
        text_coord = self.text_coord_mlp(t_feat)

        sec_emb = self.section_embedding(section_id.long())
        tt_emb = self.token_type_embedding(token_type_id.long())

        # ---- Query ----------------------------------------------------------
        if timestep_emb is None:
            t_emb = torch.zeros(B, T_a, 128, device=device, dtype=dtype)
        elif timestep_emb.dim() == 2:
            t_emb = timestep_emb.unsqueeze(1).expand(-1, T_a, -1)
        else:
            t_emb = timestep_emb

        q_in = torch.cat([pm_state, audio_coord, t_emb], dim=-1)
        q_r = self.q_mlp(q_in)

        # ---- Key / Value ----------------------------------------------------
        k_in = torch.cat([text_hidden, text_coord, sec_emb, tt_emb], dim=-1)
        k_r = self.k_mlp(k_in)
        v_r = self.v_mlp(text_hidden)

        # ---- L2-normalise q/k ----------------------------------------------
        q_r = torch.nn.functional.normalize(q_r, dim=-1, p=2)
        k_r = torch.nn.functional.normalize(k_r, dim=-1, p=2)

        # ---- Score with learnable scale ------------------------------------
        score_qk = torch.matmul(q_r, k_r.transpose(-1, -2))
        score = self.qk_score_scale * score_qk

        # ---- V4: Phase bias (dynamic, from PhaseScaffoldWarp) --------------
        phase_bias = torch.zeros_like(score)
        phase_warp_diag = {}
        _use_pb = use_phase_bias if use_phase_bias is not None else self.use_phase_bias
        if _use_pb:
            pb, p_phase, delta_p, phase_warp_diag = self.phase_scaffold_warp(
                pm_state, p_audio, c_text, timestep_emb=t_emb,
            )
            phase_bias = pb
            score = score + self.phase_bias_lambda * phase_bias

        # ---- V3: Fixed scaffold prior (fallback, off by default in v4) -----
        scaffold_prior = torch.zeros_like(score)
        prior_enabled = False
        _use_prior = use_scaffold_prior if use_scaffold_prior is not None else self.use_adapter_scaffold_prior
        if _use_prior and not _use_pb:
            dist = p_audio[:, :, None] - c_text[:, None, :]
            scaffold_prior = -(dist / self.adapter_prior_sigma) ** 2
            scaffold_prior = scaffold_prior.clamp(min=self.adapter_prior_clamp_min, max=0.0)
            if self.training and self.adapter_prior_dropout > 0:
                prior_enabled = torch.rand((), device=score.device) >= self.adapter_prior_dropout
            else:
                prior_enabled = True
            if prior_enabled:
                score = score + self.adapter_prior_lambda * scaffold_prior

        if attention_mask is not None:
            score = score.masked_fill(~attention_mask.unsqueeze(1).bool(), -1e4)

        attn_r = torch.softmax(score, dim=-1)
        ctx_r = torch.matmul(attn_r, v_r)

        # ---- Residual output ------------------------------------------------
        raw_res = self.out_proj(ctx_r)  # [B, T, D]

        if self.use_rms_writer:
            # RMS-calibrated writer
            raw_rms = torch.sqrt(raw_res.pow(2).mean(dim=-1, keepdim=True) + self.writer_eps)
            unit_res = raw_res / raw_rms
            h_rms = torch.sqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True)).detach()
            wa = self.write_alpha
            delta_h = wa * h_rms * unit_res
            retrieval_residual = delta_h
        else:
            # Fallback old writer
            retrieval_residual = self.residual_scale * torch.tanh(raw_res)

        # ---- Diagnostics ----------------------------------------------------
        with torch.no_grad():
            center = (attn_r * c_text[:, None, :]).sum(dim=-1)
            diag = {
                "pm_state_norm_before": pm_state_norm_before.item(),
                "pm_state_norm_after": pm_state_norm_after.item(),
                "score_qk_std": score_qk.std().item(),
                "score_total_std": score.std().item(),
                "attn_entropy": (-(attn_r * (attn_r + 1e-8).log()).sum(dim=-1).mean()).item(),
                "attn_max": attn_r.max(dim=-1).values.mean().item(),
                "center_mean": center.mean().item(),
                "center_std": center.std().item(),
                "delta_center_abs_mean": (center - p_audio).abs().mean().item(),
                "center_p_audio_corr": _corrcoef(center, p_audio).item(),
            }

            # Current bias diagnostics
            if _use_pb:
                diag["phase_bias_std"] = phase_bias.std().item()
                diag["qk_phase_ratio"] = (score_qk.std() / (phase_bias.std() + 1e-8)).item()
                diag["delta_p_mean"] = phase_warp_diag.get("delta_p_mean", 0)
                diag["delta_p_abs_mean"] = phase_warp_diag.get("delta_p_abs_mean", 0)
                diag["delta_p_max_abs"] = phase_warp_diag.get("delta_p_max_abs", 0)
                diag["phase_bias_enabled"] = phase_warp_diag.get("phase_bias_enabled", 0)
                diag["coef_norm"] = phase_warp_diag.get("coef_norm", 0)
            else:
                diag["score_prior_std"] = scaffold_prior.std().item()
                diag["prior_enabled"] = float(prior_enabled)

            # Writer diagnostics
            if self.use_rms_writer:
                diag["write_alpha"] = self.write_alpha.item()
                diag["hidden_norm"] = hidden_states.norm(dim=-1).mean().item()
                diag["raw_res_norm"] = raw_res.norm(dim=-1).mean().item()
                diag["delta_h_norm"] = delta_h.norm(dim=-1).mean().item()
                diag["write_ratio"] = (delta_h.norm(dim=-1).mean() / (hidden_states.norm(dim=-1).mean() + 1e-8)).item()
                diag["raw_res_rms_mean"] = raw_rms.mean().item()
                diag["unit_res_norm"] = unit_res.norm(dim=-1).mean().item()
            else:
                diag["gamma_r"] = self.gamma_r.item()
                diag["retrieval_residual_norm"] = retrieval_residual.norm(dim=-1).mean().item()

        return retrieval_residual, attn_r, diag


# ===================================================================
#  Log-domain Sinkhorn transport
# ===================================================================


def log_sinkhorn(
    log_P: torch.Tensor,
    row_mass: torch.Tensor,
    col_mass: torch.Tensor,
    iters: int = 5,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Log-domain Sinkhorn-Knopp iteration for optimal transport.

    Given a log-scale transport plan ``log_P``, iteratively normalises
    rows and columns in log space to satisfy marginal constraints.

    Args:
        log_P: [B, T, K] log-scale transport scores.
        row_mass: [B, T] target row sums (``nu``), must sum to 1 over T.
        col_mass: [B, K] target column sums (``mu``), must sum to 1 over K.
        iters: Number of Sinkhorn iterations.
        mask: [B, K] bool, ``True`` for valid columns.  Masked columns
            receive zero mass.

    Returns:
        Pi: [B, T, K] transport plan (non-negative, satisfies marginals).
        info: dict with ``row_error``, ``col_error``, ``entropy``.
    """
    B, T, K = log_P.shape
    device = log_P.device

    log_nu = torch.log(row_mass.clamp(min=1e-30))
    log_mu = torch.log(col_mass.clamp(min=1e-30))

    # Mask: set invalid columns to -inf (no mass), zero their target mass
    if mask is not None:
        log_P = log_P.masked_fill(~mask.unsqueeze(1), float("-inf"))
        # Zero out masked columns' target mass so Sinkhorn doesn't chase it
        col_mass = col_mass * mask.float()
        log_mu = torch.log(col_mass.clamp(min=1e-30))

    for _ in range(iters):
        # Row norm: log_P_ij -= logsumexp_j(log_P_ij) - log(nu_i)
        log_P = log_P - (torch.logsumexp(log_P, dim=-1, keepdim=True) - log_nu.unsqueeze(-1))
        # Col norm: log_P_ij -= logsumexp_i(log_P_ij) - log(mu_j)
        log_P = log_P - (torch.logsumexp(log_P, dim=-2, keepdim=True) - log_mu.unsqueeze(-2))
        # Re-apply mask after col norm to restore -inf (col norm on masked
        # columns produces NaN from -inf - (-inf)).
        if mask is not None:
            log_P = log_P.masked_fill(~mask.unsqueeze(1), float("-inf"))

    Pi = torch.exp(log_P)
    # Safety: zero out any remaining NaN (from masked entries) and clamp
    Pi = torch.nan_to_num(Pi, nan=0.0)
    Pi = Pi.clamp(min=0.0, max=1.0)

    with torch.no_grad():
        row_error = (Pi.sum(dim=-1) - row_mass).abs().mean().item()
        col_error = (Pi.sum(dim=-2) - col_mass).abs().mean().item()
        entropy_val = (-Pi * (Pi + 1e-10).log()).sum(dim=-1).mean().item()
        # Only check Pi for NaN; log_P may have -inf for masked columns (valid)
        has_nan = float(not torch.isfinite(Pi).all())

    info: Dict[str, float] = {
        "row_error": float(row_error),
        "col_error": float(col_error),
        "entropy": float(entropy_val),
        "sinkhorn_has_nan": has_nan,
    }
    return Pi, info


# ===================================================================
#  TransportRetrievalAdapter — unit-level Sinkhorn transport retrieval
# ===================================================================


class TransportRetrievalAdapter(nn.Module):
    """Unit-level Sinkhorn transport retrieval adapter.

    Replaces token-level softmax retrieval with **lyric-unit-level**
    Sinkhorn transport that respects both audio-side and lyric-side
    marginal constraints (``nu`` and ``mu``).

    Design:
      - Pool text hidden per lyric unit → ``unit_text_hidden [B, K, D]``.
      - Base cost from position distance ``C = (p_audio - c_unit) / sigma)²``.
      - Dynamic residual score ``R = q @ k^T / sqrt(d_r)``.
      - Combined transport logit ``L = -C + qk_scale * R``.
      - Log-domain Sinkhorn → ``Pi`` respecting ``nu`` and ``mu``.
      - Context ``ctx = Pi @ unit_value / nu`` → RMS-calibrated writer.
      - No phase bias, no fixed scaffold prior, no gamma_r gate.
    """

    def __init__(
        self,
        hidden_dim: int,
        text_dim: int,
        pm_dim: int,
        d_r: int = 64,
        coord_dim: int = 64,
        time_dim: int = 128,
        num_sections: int = 8,

        # Sinkhorn transport params
        sinkhorn_iters: int = 5,
        transport_sigma: float = 0.18,
        transport_qk_scale: float = 1.0,

        # RMS writer params
        write_alpha_init: float = 1e-4,
        write_alpha_max: float = 1e-3,
        writer_eps: float = 1e-6,
    ):
        super().__init__()
        self.d_r = d_r
        self.time_dim = time_dim
        self.sinkhorn_iters = sinkhorn_iters
        self.transport_sigma = transport_sigma
        self.transport_qk_scale = transport_qk_scale
        self.write_alpha_max = write_alpha_max
        self.writer_eps = writer_eps

        # PM state LayerNorm
        self.pm_state_norm = nn.LayerNorm(pm_dim)

        # Audio-side coordinate MLP
        self.audio_coord_mlp = nn.Sequential(
            nn.Linear(3, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
            nn.SiLU(),
        )

        # Unit-side coordinate MLP
        self.unit_coord_mlp = nn.Sequential(
            nn.Linear(3, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
            nn.SiLU(),
        )

        # Section embedding (for lyric unit section)
        self.section_embedding = nn.Embedding(num_sections, coord_dim // 2)

        # Query MLP (audio → retrieval query)
        q_in_dim = pm_dim + coord_dim + time_dim
        self.q_mlp = nn.Sequential(
            nn.Linear(q_in_dim, d_r * 2),
            nn.SiLU(),
            nn.Linear(d_r * 2, d_r),
        )

        # Key MLP (unit text → retrieval key)
        k_in_dim = text_dim + coord_dim + coord_dim // 2
        self.k_mlp = nn.Sequential(
            nn.Linear(k_in_dim, d_r * 2),
            nn.SiLU(),
            nn.Linear(d_r * 2, d_r),
        )

        # Value MLP (unit text → retrieval value)
        self.v_mlp = nn.Sequential(
            nn.Linear(text_dim, d_r * 2),
            nn.SiLU(),
            nn.Linear(d_r * 2, d_r),
        )

        # Output projection
        self.out_proj = nn.Linear(d_r, hidden_dim)
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

        # RMS writer alpha (learnable fraction of h_rms to write)
        init_prob = max(write_alpha_init / max(write_alpha_max, 1e-10), 1e-6)
        init_prob = min(init_prob, 0.999)
        self.write_logit = nn.Parameter(torch.tensor(math.log(init_prob / (1.0 - init_prob))))

    @property
    def write_alpha(self) -> torch.Tensor:
        return self.write_alpha_max * torch.sigmoid(self.write_logit)

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_hidden: torch.Tensor,
        pm_state: torch.Tensor,
        p_audio: torch.Tensor,
        unit_text_hidden: torch.Tensor,
        unit_c_pos: torch.Tensor,
        unit_mass: torch.Tensor,
        unit_section_id: Optional[torch.Tensor] = None,
        timestep_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Args:
            hidden_states: [B, T, D] layer-12 hidden states.
            text_hidden: [B, L, D] unused — kept for compat signature.
            pm_state: [B, T, pm_dim] phase state (will be LayerNorm'd).
            p_audio: [B, T] audio progress in [0, 1].
            unit_text_hidden: [B, K, D] pooled text hidden per lyric unit.
            unit_c_pos: [B, K] unit centre positions in [0, 1].
            unit_mass: [B, K] unit mass ``mu``, sum=1 over K.
            unit_section_id: [B, K] optional section ID per unit.
            timestep_emb: [B, T, time_dim] or None.

        Returns:
            delta_h: [B, T, D] RMS-calibrated hidden residual.
            Pi: [B, T, K] Sinkhorn transport plan.
            diag: dict of diagnostics.
        """
        B, T_a = p_audio.shape
        K = unit_c_pos.shape[-1]
        device = p_audio.device
        dtype = p_audio.dtype

        if K == 0:
            return torch.zeros_like(hidden_states), torch.zeros(B, T_a, 0, device=device), {}

        # ---- 1. Query ------------------------------------------------------------
        pm_state = self.pm_state_norm(pm_state)

        a_feat = torch.stack([p_audio, p_audio ** 2, 1.0 - p_audio], dim=-1)
        audio_coord = self.audio_coord_mlp(a_feat)

        if timestep_emb is None:
            t_emb = torch.zeros(B, T_a, self.time_dim, device=device, dtype=dtype)
        elif timestep_emb.dim() == 2:
            t_emb = timestep_emb.unsqueeze(1).expand(-1, T_a, -1)
        else:
            t_emb = timestep_emb

        q_in = torch.cat([pm_state, audio_coord, t_emb], dim=-1)
        q = F.normalize(self.q_mlp(q_in), dim=-1, p=2)

        # ---- 2. Key / Value ------------------------------------------------------
        u_feat = torch.stack([unit_c_pos, unit_c_pos ** 2, 1.0 - unit_c_pos], dim=-1)
        unit_coord = self.unit_coord_mlp(u_feat)

        sec_emb = self.section_embedding(unit_section_id.long()) if unit_section_id is not None \
                  else torch.zeros(B, K, self.section_embedding.embedding_dim, device=device, dtype=dtype)

        k_in = torch.cat([unit_text_hidden, unit_coord, sec_emb], dim=-1)
        k = F.normalize(self.k_mlp(k_in), dim=-1, p=2)
        v = self.v_mlp(unit_text_hidden)

        # ---- 3. Base cost (position distance) ------------------------------------
        dist = p_audio[:, :, None] - unit_c_pos[:, None, :]  # [B, T, K]
        C = (dist / self.transport_sigma) ** 2
        base_logit = -C  # negative cost: closer = higher logit

        # ---- 4. Dynamic residual score ------------------------------------------
        R = torch.matmul(q, k.transpose(-1, -2))  # [B, T, K]
        R = R / math.sqrt(self.d_r)

        # ---- 5. Combined transport logit -----------------------------------------
        L = base_logit + self.transport_qk_scale * R

        # ---- 6. Row mass (uniform over audio timesteps) --------------------------
        nu = torch.full((B, T_a,), 1.0 / T_a, device=device, dtype=dtype)

        # ---- 7. Log-domain Sinkhorn ----------------------------------------------
        Pi, sinkhorn_info = log_sinkhorn(
            L, nu, unit_mass,
            iters=self.sinkhorn_iters,
        )

        # ---- 8. Context (per-audio-token, dividing by row mass) ------------------
        ctx = torch.matmul(Pi, v)  # [B, T, d_r]
        ctx = ctx / nu.unsqueeze(-1).clamp(min=1e-10)

        # ---- 9. RMS-calibrated writer -------------------------------------------
        raw_res = self.out_proj(ctx)  # [B, T, D]
        raw_rms = torch.sqrt(raw_res.pow(2).mean(dim=-1, keepdim=True) + self.writer_eps)
        unit_res = raw_res / raw_rms
        h_rms = torch.sqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True)).detach()
        wa = self.write_alpha
        delta_h = wa * h_rms * unit_res

        # ---- 10. Diagnostics ------------------------------------------------------
        with torch.no_grad():
            diag: Dict[str, float] = dict(sinkhorn_info)
            diag["transport_max"] = Pi.max().item()
            diag["transport_mean"] = Pi.mean().item()
            diag["qk_std"] = R.std().item()
            diag["base_logit_std"] = base_logit.std().item()
            diag["transport_qk_ratio"] = (R.std() / (base_logit.std() + 1e-8)).item()
            diag["write_alpha"] = wa.item()
            diag["write_ratio"] = (delta_h.norm(dim=-1).mean() / (hidden_states.norm(dim=-1).mean() + 1e-8)).item()
            diag["hidden_delta_norm"] = delta_h.norm(dim=-1).mean().item()
            diag["raw_res_norm"] = raw_res.norm(dim=-1).mean().item()
            diag["unit_mass_min"] = unit_mass.min().item() if K > 0 else 0.0
            diag["unit_mass_max"] = unit_mass.max().item() if K > 0 else 0.0
            diag["final_output_delta_ratio"] = diag["write_ratio"]
            diag["has_nan"] = float(not (torch.isfinite(delta_h).all() and torch.isfinite(Pi).all() and torch.isfinite(raw_res).all()))

        return delta_h, Pi, diag


# ===================================================================
#  Scaffold progress helpers
# ===================================================================

def scaffold_progress(
    scaffold: dict,
    T_audio: int,
    device: Optional[torch.device] = None,
    batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract p_audio and c_text from a duration scaffold.

    Args:
        scaffold: Output of ``build_duration_scaffold``.
        T_audio: Number of audio timesteps.
        device: Torch device.
        batch_size: If > 0, expand all outputs to [B, T] / [B, L].

    Returns:
        p_audio: [T_audio] or [B, T_audio] linear progress in [0, 1].
        c_text: [L] or [B, L] unit centre position (lyric + control tokens).
        token_type_id: [L] or [B, L] 0=lyric, 1=section_tag, 2=control, 3=other.
    """
    # Normalise scaffold tensors to CPU numpy for indexing
    def _np(t):
        if isinstance(t, torch.Tensor):
            return t.cpu().numpy()
        return np.asarray(t)

    u_b = _np(scaffold["unit_boundaries"])   # [U+1]
    t2u = _np(scaffold["token_to_unit"])     # [L]
    lm = _np(scaffold["lyric_mask"])          # [L] bool
    cm = _np(scaffold.get("control_mask", np.zeros_like(lm, dtype=bool)))

    L = len(lm)
    u_centers = (u_b[:-1] + u_b[1:]) / 2     # [U]

    c_text = np.zeros(L, dtype=np.float32)
    ttid = np.zeros(L, dtype=np.int32)

    for j in range(L):
        if lm[j]:
            uid = int(t2u[j])
            c_text[j] = u_centers[min(uid, len(u_centers) - 1)]
            ttid[j] = 0  # lyric
        elif cm[j]:
            # Control tokens also get their unit centre
            uid = int(t2u[j])
            if uid >= 0:
                c_text[j] = u_centers[min(uid, len(u_centers) - 1)]
            ttid[j] = 1  # section/control tag
        else:
            ttid[j] = 3  # other

    p_audio = torch.linspace(0, 1, T_audio, device=device, dtype=torch.float32)
    c_text_t = torch.from_numpy(c_text).to(device).float()
    ttid_t = torch.from_numpy(ttid).to(device).long()

    if batch_size > 0:
        p_audio = p_audio.unsqueeze(0).expand(batch_size, -1)
        c_text_t = c_text_t.unsqueeze(0).expand(batch_size, -1)
        ttid_t = ttid_t.unsqueeze(0).expand(batch_size, -1)

    return p_audio, c_text_t, ttid_t


# ===================================================================
#  Backward compatibility aliases for old model code
# ===================================================================
# NOTE: PhaseMemory is the OLD hidden-injector PM for adapter_type="phase_memory".
#       Do NOT use in new code — use PhaseMemoryDurationClock, PMDCResidualClock,
#       or PMRetrievalPhaseMemory explicitly.
import logging as _pm_logging
_pm_logging.getLogger(__name__).warning(
    "[DEPRECATED] PhaseMemory alias imported — use PhaseMemoryDurationClock or "
    "PMRetrievalPhaseMemory explicitly in new code."
)
PhaseMemory = PhaseMemoryDurationClock

if __name__ == "__main__":
    smoke_test_pmdc()
