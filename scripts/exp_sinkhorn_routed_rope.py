#!/usr/bin/env python3
"""
Sinkhorn-Routed Residual RoPE — inference-time position warping for ACE-Step 1.5.

Replaces the uniform position layout in DiT self-attention with a structure-aware
layout induced by a Sinkhorn coupling between audio tokens and lyric/structure units.

Modes
-----
  baseline       — no Sinkhorn, no RoPE correction (vanilla generation)
  sinkhorn_only  — compute Sinkhorn coupling, output diagnostics, no RoPE change
  rope_only      — heuristic structure prior RoPE (no Sinkhorn coupling)
  sinkhorn_rope  — full Sinkhorn coupling + Residual RoPE correction

Usage
-----
  python scripts/exp_sinkhorn_routed_rope.py --mode sinkhorn_rope --duration 60
"""

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

# Default model path — override with --model-root
_DEFAULT_MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# ---------------------------------------------------------------------------
# 1.  Structure unit parser
# ---------------------------------------------------------------------------

SECTION_PATTERN = re.compile(r"^\[(\w+)\]$")
NON_LYRIC_SECTIONS = {"intro", "instrumental", "outro", "silence", "interlude"}


def parse_lyrics_to_units(lyrics: str) -> List[Dict[str, Any]]:
    """Parse lyrics text into a list of structure units.

    Rules
    -----
    * ``[SectionTag]`` lines are treated as section boundaries.
    * Non-empty lines under a section become lyric units belonging to that section.
    * Non-lyric sections (Intro, Instrumental, Outro, Silence, Interlude) are
      kept as standalone units with their section tag name.
    * When no section tag is present, each non-empty line is one unit.

    Returns
    -------
    List[Dict] with keys: name, type, text.
    """
    units: List[Dict[str, Any]] = []
    lines = lyrics.strip().split("\n")
    current_section: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        m = SECTION_PATTERN.match(stripped)
        if m:
            tag = m.group(1).lower()
            current_section = m.group(1)
            # Non-lyric sections become standalone units
            if tag in NON_LYRIC_SECTIONS:
                units.append({
                    "name": current_section,
                    "type": "non_lyric",
                    "text": f"[{current_section}]",
                })
            continue
        # Lyric line
        units.append({
            "name": current_section or "lyric",
            "type": "lyric",
            "text": stripped,
        })

    # If nothing was parsed, fall back to treating the whole input as one unit
    if not units:
        units.append({
            "name": "content",
            "type": "lyric",
            "text": lyrics.strip(),
        })
    return units


# ---------------------------------------------------------------------------
# 2.  Budget allocation
# ---------------------------------------------------------------------------

_HEURISTIC_BUDGET = {
    "intro": 0.08,
    "outro": 0.08,
    "instrumental": 0.10,
    "silence": 0.04,
    "interlude": 0.06,
}


def allocate_budgets(units: List[Dict[str, Any]], gamma: float = 0.2) -> torch.Tensor:
    """Allocate a normalised budget vector over units.

    Heuristic
    --------
    * Reserved non-lyric sections get a fixed proportion.
    * The remainder is spread equally over lyric-type units.
    * Budget smoothing:  b' = (1-γ)b + γ/U

    Returns
    -------
    Tensor ``[1, U]``, sum = 1.
    """
    U = len(units)
    budgets = np.zeros(U, dtype=np.float64)

    # 1. Assign reserved budgets
    reserved = 0.0
    for i, u in enumerate(units):
        key = u["name"].lower()
        if u["type"] == "non_lyric" and key in _HEURISTIC_BUDGET:
            budgets[i] = _HEURISTIC_BUDGET[key]
            reserved += budgets[i]

    # 2. Distribute remainder over lyric units
    lyric_indices = [i for i, u in enumerate(units) if u["type"] == "lyric"]
    if lyric_indices:
        remainder = max(0.0, 1.0 - reserved)
        per_unit = remainder / len(lyric_indices)
        for i in lyric_indices:
            budgets[i] = per_unit
    elif reserved < 1.0:
        # No lyric units: spread remainder over all
        remainder = 1.0 - reserved
        for i in range(U):
            budgets[i] += remainder / U

    # 3. Smoothing
    budgets = (1.0 - gamma) * budgets + gamma / U

    # 4. Normalise
    budgets = budgets / budgets.sum()
    return torch.from_numpy(budgets).float().unsqueeze(0)  # [1, U]


def budgets_to_info(units: List[Dict], budgets: torch.Tensor) -> List[Dict]:
    """Attach budget values back to units for diagnostics."""
    info = []
    for i, u in enumerate(units):
        info.append({
            "name": u["name"],
            "type": u["type"],
            "text": u["text"],
            "budget": round(float(budgets[0, i]), 6),
        })
    return info


# ---------------------------------------------------------------------------
# 3.  Sinkhorn coupling
# ---------------------------------------------------------------------------


def sinkhorn(
    K: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    iters: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-domain-stable Sinkhorn algorithm (standard domain).

    Args:
        K:  [B, N, U]  — exponentiated negative cost (exp(-C/ε)).
        a:  [B, N]     — row marginal constraint.
        b:  [B, U]     — column marginal constraint.
    Returns:
        P:  [B, N, U]  — coupling matrix.
    """
    u = torch.ones_like(a)          # [B, N]
    v = torch.ones_like(b)          # [B, U]
    for _ in range(iters):
        u = a / ((K @ v.unsqueeze(-1)).squeeze(-1) + eps)   # [B, N]
        v = b / ((K.transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1) + eps)  # [B, U]
    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)                # [B, N, U]
    return P


def compute_sinkhorn_warped_positions(
    seq_len: int,
    budgets: torch.Tensor,          # [1, U]
    rho: torch.Tensor,              # [1, U]
    tau: torch.Tensor,              # [1, N]
    beta: float = 1.0,
    max_token_shift: float = 0.25,
    sinkhorn_iters: int = 5,
    sinkhorn_epsilon: float = 0.1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Full Sinkhorn → warped position pipeline (token-space shift).

    Formula:  p'_i = p_i + clip(β·(N-1)·(μ_i − τ_i), −Δ_max, Δ_max)

    Returns
    -------
    warped_position_ids : [1, N] float tensor.
    diagnostics         : dict with coupling, shift & slope stats.
    """
    N = seq_len
    U = budgets.shape[-1]

    # Move tensors to target device
    budgets = budgets.to(device=device, dtype=dtype)
    rho = rho.to(device=device, dtype=dtype)
    tau = tau.to(device=device, dtype=dtype)

    # Distance cost:  C_{iu} = (tau_i - rho_u)^2 / (b_u + epsilon)^2
    b_safe = budgets + sinkhorn_epsilon  # [1, U]
    tau_exp = tau.unsqueeze(-1)          # [1, N, 1]
    rho_exp = rho.unsqueeze(-2)          # [1, 1, U]
    b_exp = b_safe.unsqueeze(-2)         # [1, 1, U]
    C = ((tau_exp - rho_exp) / b_exp) ** 2  # [1, N, U]

    # Affinity (Gibbs kernel)
    K = torch.exp(-C / sinkhorn_epsilon)    # [1, N, U]

    # Marginals
    a = torch.full((1, N), 1.0 / N, device=device, dtype=dtype)  # [1, N]
    b = budgets                              # [1, U]

    # Sinkhorn
    P = sinkhorn(K, a, b, iters=sinkhorn_iters, eps=1e-8)  # [1, N, U]

    # Row-normalised coupling  π_{iu}
    pi = P / (P.sum(dim=-1, keepdim=True) + 1e-8)  # [1, N, U]

    # Structural progress  μ_i = sum_u π_{iu} ρ_u
    mu = (pi * rho_exp).sum(dim=-1)  # [1, N]

    # ---- Token-space shift (new formula) ----
    p_orig = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # [1, N]
    raw_shift = beta * (N - 1) * (mu - tau)  # [1, N], token units
    delta_p = torch.clamp(raw_shift, -max_token_shift, max_token_shift)
    p_prime = p_orig + delta_p

    # Enforce monotonicity
    p_prime = torch.cummax(p_prime, dim=-1).values

    # ---- Slope diagnostics ----
    v = p_prime[0, 1:] - p_prime[0, :-1]  # [N-1]
    n_flat = int((v < 0.9).sum().item())
    n_stretched = int((v > 1.1).sum().item())

    diagnostics = {
        "budget": budgets.detach().cpu().tolist(),
        "rho": rho.detach().cpu().tolist(),
        "tau": tau.detach().cpu().tolist(),
        "mu": mu.detach().cpu().tolist(),
        "p_orig": p_orig[0].tolist(),
        "p_prime": p_prime[0].tolist(),
        "delta_p_mean": float(delta_p.mean().item()),
        "delta_p_abs_mean": float(delta_p.abs().mean().item()),
        "delta_p_abs_max": float(delta_p.abs().max().item()),
        "slope_mean": float(v.mean().item()),
        "slope_std": float(v.std().item()),
        "slope_min": float(v.min().item()),
        "slope_max": float(v.max().item()),
        "n_flat": n_flat,
        "n_stretched": n_stretched,
        "pct_flat": round(n_flat / max(N - 1, 1) * 100, 2),
        "pct_stretched": round(n_stretched / max(N - 1, 1) * 100, 2),
        "coupling_col_mass": P.sum(dim=1).detach().cpu().tolist(),
        "coupling_row_mass_mean": float(P.sum(dim=-1).mean().item()),
    }

    return p_prime, diagnostics


# ---------------------------------------------------------------------------
# 4.  Heuristic (training-free) warped positions  —  rope_only mode
# ---------------------------------------------------------------------------

def compute_heuristic_warped_positions(
    seq_len: int,
    budgets: torch.Tensor,   # [1, U]
    rho: torch.Tensor,       # [1, U]
    beta: float = 1.0,
    max_token_shift: float = 0.25,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Heuristic structural position without Sinkhorn (token-space shift).

    Uses the budget centroids directly to compute structural progress without
    learning a coupling.
    """
    N = seq_len
    U = budgets.shape[-1]

    budgets = budgets.to(device=device, dtype=dtype)
    rho = rho.to(device=device, dtype=dtype)

    # Normalised audio positions  τ_i = i / (N-1)
    tau = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype).unsqueeze(0)  # [1, N]

    # Heuristic: assign each audio token to the nearest unit centroid.
    # Distance: d_{iu} = |τ_i - ρ_u|
    tau_exp = tau.unsqueeze(-1)   # [1, N, 1]
    rho_exp = rho.unsqueeze(-2)   # [1, 1, U]
    dist = (tau_exp - rho_exp).abs()
    # Soft assignment via normalised inverse distance
    weight = 1.0 / (dist + 0.01)
    pi = weight / weight.sum(dim=-1, keepdim=True)  # [1, N, U]

    # Structural progress
    mu = (pi * rho_exp).sum(dim=-1)  # [1, N]

    # ---- Token-space shift ----
    p_orig = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # [1, N]
    raw_shift = beta * (N - 1) * (mu - tau)  # [1, N], token units
    delta_p = torch.clamp(raw_shift, -max_token_shift, max_token_shift)
    p_prime = p_orig + delta_p
    p_prime = torch.cummax(p_prime, dim=-1).values

    # ---- Slope diagnostics ----
    v = p_prime[0, 1:] - p_prime[0, :-1]
    n_flat = int((v < 0.9).sum().item())
    n_stretched = int((v > 1.1).sum().item())

    diagnostics = {
        "budget": budgets.detach().cpu().tolist(),
        "rho": rho.detach().cpu().tolist(),
        "tau": tau.detach().cpu().tolist(),
        "mu": mu.detach().cpu().tolist(),
        "p_orig": p_orig[0].tolist(),
        "p_prime": p_prime[0].tolist(),
        "delta_p_mean": float(delta_p.mean().item()),
        "delta_p_abs_mean": float(delta_p.abs().mean().item()),
        "delta_p_abs_max": float(delta_p.abs().max().item()),
        "slope_mean": float(v.mean().item()),
        "slope_std": float(v.std().item()),
        "slope_min": float(v.min().item()),
        "slope_max": float(v.max().item()),
        "n_flat": n_flat,
        "n_stretched": n_stretched,
        "pct_flat": round(n_flat / max(N - 1, 1) * 100, 2),
        "pct_stretched": round(n_stretched / max(N - 1, 1) * 100, 2),
    }

    return p_prime, diagnostics


# ---------------------------------------------------------------------------
# 5.  Sinkhorn RoPE context and model patching
# ---------------------------------------------------------------------------

class SinkhornRopeContext:
    """Holds all state required for the monkey-patched Sinkhorn RoPE correction."""

    def __init__(
        self,
        model: Any,
        lyrics: str,
        mode: str = "sinkhorn_rope",
        rope_layers: Optional[List[int]] = None,
        rope_start_ratio: float = 0.3,
        rope_end_ratio: float = 0.7,
        beta: float = 1.0,
        max_token_shift: float = 0.25,
        full_attention_only: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 0.1,
        budget_smoothing_gamma: float = 0.2,
        total_steps: int = 25,
    ):
        self.model = model
        self.lyrics = lyrics
        self.mode = mode
        self.rope_layers = rope_layers or []
        self.rope_start_ratio = rope_start_ratio
        self.rope_end_ratio = rope_end_ratio
        self.beta = beta
        self.max_token_shift = max_token_shift
        self.full_attention_only = full_attention_only
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.budget_smoothing_gamma = budget_smoothing_gamma
        self.total_steps = total_steps

        # Runtime state (start at -1 so first increment → step 0)
        self.diff_step = -1
        self.warped_position_cache: Dict[int, torch.Tensor] = {}
        self.diagnostics_cache: Dict[int, Dict] = {}
        # Track whether patches have been applied (to avoid double-apply)
        self._patches_applied = False

        # Target sequence length (set on first call)
        self._seq_len: Optional[int] = None

        # Parse lyrics
        self.units = parse_lyrics_to_units(lyrics)

        # Budgets and centroids
        self.budgets = allocate_budgets(self.units, gamma=budget_smoothing_gamma)
        self.rho = self._compute_centroids(self.budgets)

    def _compute_centroids(self, budgets: torch.Tensor) -> torch.Tensor:
        """Cumulative centre-of-mass:  ρ_u = Σ_{v<u} b_v + b_u / 2."""
        cumsum = budgets.cumsum(dim=-1)  # [1, U]
        rho = cumsum - 0.5 * budgets     # [1, U]
        return rho

    def should_use(self, layer_idx: int) -> bool:
        """Check if the current diffusion step and layer should use warped RoPE."""
        if self.mode == "baseline":
            return False
        if layer_idx not in self.rope_layers:
            return False
        if self.full_attention_only:
            layer = self.model.decoder.layers[layer_idx]
            if layer.attention_type != "full_attention":
                return False
        if self.total_steps <= 1:
            return False
        r = self.diff_step / max(self.total_steps - 1, 1)
        return self.rope_start_ratio <= r <= self.rope_end_ratio

    def get_warped_positions(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return cached or freshly computed warped position IDs.

        Returns
        -------
        Tensor [1, seq_len] on the requested device.
        """
        if seq_len in self.warped_position_cache:
            return self.warped_position_cache[seq_len].to(device=device, dtype=dtype)

        N = seq_len
        tau = torch.linspace(0.0, 1.0, N, device="cpu", dtype=torch.float32).unsqueeze(0)

        if self.mode in ("sinkhorn_rope", "sinkhorn_only"):
            warped_ids, diag = compute_sinkhorn_warped_positions(
                seq_len=N,
                budgets=self.budgets.clone(),
                rho=self.rho.clone(),
                tau=tau,
                beta=self.beta,
                max_token_shift=self.max_token_shift,
                sinkhorn_iters=self.sinkhorn_iters,
                sinkhorn_epsilon=self.sinkhorn_epsilon,
                device="cpu",
                dtype=torch.float32,
            )
        elif self.mode == "rope_only":
            warped_ids, diag = compute_heuristic_warped_positions(
                seq_len=N,
                budgets=self.budgets.clone(),
                rho=self.rho.clone(),
                beta=self.beta,
                max_token_shift=self.max_token_shift,
                device="cpu",
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.warped_position_cache[seq_len] = warped_ids
        self.diagnostics_cache[seq_len] = diag
        return warped_ids.to(device=device, dtype=dtype)

    def summary(self) -> Dict[str, Any]:
        """Return a serialisable summary of the context state."""
        first_diag = None
        for d in self.diagnostics_cache.values():
            first_diag = d
            break

        unit_info = budgets_to_info(self.units, self.budgets)

        slope_info = {}
        for d in self.diagnostics_cache.values():
            slope_info = {
                "delta_p_abs_mean": d.get("delta_p_abs_mean"),
                "delta_p_abs_max": d.get("delta_p_abs_max"),
                "slope_mean": d.get("slope_mean"),
                "slope_min": d.get("slope_min"),
                "slope_max": d.get("slope_max"),
                "pct_flat": d.get("pct_flat"),
                "pct_stretched": d.get("pct_stretched"),
            }
            break

        return {
            "use_sinkhorn_rope": self.mode in ("sinkhorn_rope", "sinkhorn_only"),
            "beta": self.beta,
            "max_token_shift": self.max_token_shift,
            "full_attention_only": self.full_attention_only,
            "rope_layers": self.rope_layers,
            "rope_start_ratio": self.rope_start_ratio,
            "rope_end_ratio": self.rope_end_ratio,
            "sinkhorn_iters": self.sinkhorn_iters,
            "sinkhorn_epsilon": self.sinkhorn_epsilon,
            "budget_smoothing_gamma": self.budget_smoothing_gamma,
            "num_units": len(self.units),
            "units": unit_info,
            "budget": self.budgets[0].tolist(),
            **slope_info,
        }


def apply_sinkhorn_rope_patches(ctx: SinkhornRopeContext):
    """Monkey-patch the model to apply Sinkhorn-Routed Residual RoPE.

    Patches:
        1. Decoder ``forward`` — counts diffusion steps.
        2. Enabled layer ``forward`` — replaces ``position_embeddings`` with
           warped versions when the layer and diffusion step qualify.

    Idempotent: safe to call multiple times (no-op after first apply).
    """
    if ctx._patches_applied:
        return
    ctx._patches_applied = True

    decoder = ctx.model.decoder

    # --- 1. Patch decoder.forward to count diffusion steps ---
    orig_decoder_fwd = decoder.forward

    def patched_decoder_fwd(*args, **kwargs):
        ctx.diff_step += 1
        return orig_decoder_fwd(*args, **kwargs)

    decoder.forward = patched_decoder_fwd

    # --- 2. Patch enabled layers to swap position_embeddings ---
    for layer_idx in ctx.rope_layers:
        if layer_idx >= len(decoder.layers):
            continue
        layer = decoder.layers[layer_idx]
        orig_layer_fwd = layer.forward

        def make_patched_layer_forward(l_idx: int, orig_fwd):
            def patched_layer_fwd(
                hidden_states,
                position_embeddings,
                temb,
                diffusion_step=None,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                **kwargs,
            ):
                if ctx.should_use(l_idx):
                    # Compute warped position embeddings at target sequence length
                    B, T = hidden_states.shape[0], hidden_states.shape[1]
                    device = hidden_states.device
                    dtype = hidden_states.dtype
                    warped_pos = ctx.get_warped_positions(T, device, dtype)
                    # Recompute cos/sin with warped positions
                    new_cos, new_sin = ctx.model.decoder.rotary_emb(
                        hidden_states, warped_pos
                    )
                    position_embeddings = (new_cos, new_sin)

                return orig_fwd(
                    hidden_states,
                    position_embeddings,
                    temb,
                    diffusion_step=diffusion_step,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    **kwargs,
                )

            return patched_layer_fwd

        layer.forward = make_patched_layer_forward(layer_idx, orig_layer_fwd)


# ---------------------------------------------------------------------------
# 6.  Generation invocation
# ---------------------------------------------------------------------------

DEFAULT_LYRICS = """[Intro]

[Verse]
漫天的星光 照亮了夜晚
微风轻轻吹 带来你的温暖
走过的路上 花开又花落
每一刻都是 最美的时光

[Chorus]
你就是我心中 最亮的光
照亮所有黑暗 不再迷茫
不管未来多远 路有多长
握紧你的手 一起飞翔

[Verse]
城市的灯火 闪烁着希望
你的微笑 是我最暖的阳光
风雨中前行 有你陪在身旁
每一天都是 最美的篇章

[Chorus]
你就是我心中 最亮的光
照亮所有黑暗 不再迷茫
不管未来多远 路有多长
握紧你的手 一起飞翔

[Bridge]
时光流转 不会改变
这份爱永远 在心间
就算世界 沧海桑田
你依然是我 最亮的星

[Chorus]
你就是我心中 最亮的光
照亮所有黑暗 不再迷茫
不管未来多远 路有多长
握紧你的手 一起飞翔

[Outro]"""


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run one generation experiment and return results + diagnostics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp_sinkhorn_routed_rope] Device: {device}")
    print(f"[exp_sinkhorn_routed_rope] Mode: {args.mode}")

    # ---- 1. Init handler ------------------------------------------------
    print("[exp_sinkhorn_routed_rope] Loading ACE-Step handler ...")
    dit = AceStepHandler()
    status_msg, ok = dit.initialize_service(
        project_root=str(args.model_root),
        config_path=args.config,
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not ok:
        print(f"[exp_sinkhorn_routed_rope] Handler init FAILED: {status_msg}")
        return {"error": f"Handler init failed: {status_msg}"}
    print(f"[exp_sinkhorn_routed_rope] Handler init: {status_msg}")

    model = dit.model.eval()

    # Disable PhaseMemory if present (not relevant for this experiment)
    for l in model.decoder.layers:
        if getattr(l, "use_phase_memory", False):
            l.use_phase_memory = False

    print("[exp_sinkhorn_routed_rope] Initializing LLM handler ...")
    llm = LLMHandler()
    _, llm_ok = llm.initialize(
        checkpoint_dir=str(args.model_root),
        lm_model_path=args.lm_model,
        backend="pt",
        device="cuda",
    )
    if not llm_ok:
        print("[exp_sinkhorn_routed_rope] LLM init FAILED; disabling thinking")
        args.thinking = False
        args.use_cot_metas = False
        args.use_cot_caption = False

    lyrics_text = args.lyrics if args.lyrics else DEFAULT_LYRICS

    # ---- 2. Create context and optionally apply patches -----------------
    ctx = None
    if args.mode in ("sinkhorn_rope", "rope_only", "sinkhorn_only"):
        rope_layers = list(args.rope_layers) if args.rope_layers else []
        if not rope_layers:
            print("[exp_sinkhorn_routed_rope] No rope-layers specified; defaulting to [8, 12, 16, 20]")
            rope_layers = [8, 12, 16, 20]

        ctx = SinkhornRopeContext(
            model=model,
            lyrics=lyrics_text,
            mode=args.mode,
            rope_layers=rope_layers,
            rope_start_ratio=args.rope_start_ratio,
            rope_end_ratio=args.rope_end_ratio,
            beta=args.rope_beta,
            max_token_shift=args.max_token_shift,
            full_attention_only=args.full_attention_only,
            sinkhorn_iters=args.sinkhorn_iters,
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            budget_smoothing_gamma=args.budget_smoothing_gamma,
            total_steps=args.steps,
        )

        if args.mode in ("sinkhorn_rope", "rope_only"):
            print(f"[exp_sinkhorn_routed_rope] Applying Sinkhorn-RoPE patches ...")
            print(f"  Layers: {ctx.rope_layers}  (full_attention_only={ctx.full_attention_only})")
            print(f"  Beta: {ctx.beta}, MaxTokenShift: {ctx.max_token_shift}")
            print(f"  Step range: {args.rope_start_ratio} – {args.rope_end_ratio}")
            print(f"  Units: {len(ctx.units)}")
            apply_sinkhorn_rope_patches(ctx)
        elif args.mode == "sinkhorn_only":
            print(f"[exp_sinkhorn_routed_rope] Sinkhorn-only mode: computing coupling diagnostics ...")
            print(f"  Units: {len(ctx.units)}, Budgets: {ctx.budgets[0].tolist()}")
            print(f"  (no RoPE patches applied)")

    # ---- 3. Build generation parameters --------------------------------
    params = GenerationParams(
        task_type="text2music",
        caption=args.caption,
        lyrics=lyrics_text,
        instrumental=False,
        bpm=args.bpm,
        keyscale=args.key,
        timesignature=args.time_sig,
        vocal_language=args.vocal_lang,
        duration=args.duration,
        inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        thinking=args.thinking,
        use_cot_metas=args.use_cot_metas,
        use_cot_caption=args.use_cot_caption,
        lm_temperature=0.75,
    )
    gen_config = GenerationConfig(
        batch_size=1,
        audio_format="flac",
        use_random_seed=False,
        seeds=[args.seed],
    )

    # ---- 4. Run generation ---------------------------------------------
    print(f"[exp_sinkhorn_routed_rope] Starting generation ...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = generate_music(
            dit_handler=dit,
            llm_handler=llm,
            params=params,
            config=gen_config,
            save_dir=str(output_dir),
        )

        # Collect results
        audio_paths = []
        if result.success and result.audios:
            audio_paths = [a["path"] for a in result.audios]
            print(f"[exp_sinkhorn_routed_rope] Audio: {audio_paths[0]}")
        else:
            print(f"[exp_sinkhorn_routed_rope] Generation status: {result.status_message}")

    except Exception as exc:
        print(f"[exp_sinkhorn_routed_rope] Generation FAILED: {exc}")
        traceback.print_exc()
        result = None
        audio_paths = []

    # ---- 5. Save diagnostics -------------------------------------------
    summary_data = {
        "duration": args.duration,
        "steps": args.steps,
        "seed": args.seed,
        "mode": args.mode,
        "use_sinkhorn_rope": args.mode in ("sinkhorn_rope", "sinkhorn_only"),
        "beta": args.rope_beta,
        "max_token_shift": args.max_token_shift,
        "full_attention_only": args.full_attention_only,
        "rope_layers": list(args.rope_layers) if args.rope_layers else [],
        "rope_start_ratio": args.rope_start_ratio,
        "rope_end_ratio": args.rope_end_ratio,
        "sinkhorn_iters": args.sinkhorn_iters,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "budget_smoothing_gamma": args.budget_smoothing_gamma,
    }

    if ctx is not None:
        sm = ctx.summary()
        summary_data.update(sm)
        summary_data["num_units"] = len(ctx.units)
        summary_data["units"] = budgets_to_info(ctx.units, ctx.budgets)
        summary_data["budget"] = ctx.budgets[0].tolist()

        # Save detailed diagnostics tensor
        diag_path = output_dir / "sinkhorn_rope_diagnostics.pt"
        diag_dict = {}
        for seq_len, diag in ctx.diagnostics_cache.items():
            diag_dict[str(seq_len)] = {
                k: v for k, v in diag.items()
            }
        torch.save(diag_dict, diag_path)
        print(f"[exp_sinkhorn_routed_rope] Diagnostics saved: {diag_path}")

    summary_path = output_dir / "sinkhorn_rope_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[exp_sinkhorn_routed_rope] Summary saved: {summary_path}")

    print(f"[exp_sinkhorn_routed_rope] Done.")
    return {
        "result": result,
        "audio_paths": audio_paths,
        "summary": summary_data,
        "ctx": ctx,
    }


# ---------------------------------------------------------------------------
# 7.  CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sinkhorn-Routed Residual RoPE experiment for ACE-Step 1.5"
    )

    # Mode
    parser.add_argument("--mode", type=str, default="sinkhorn_rope",
                        choices=["baseline", "sinkhorn_only", "rope_only", "sinkhorn_rope"],
                        help="Experiment mode (default: sinkhorn_rope)")

    # Generation
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Audio duration in seconds (default: 60)")
    parser.add_argument("--steps", type=int, default=25,
                        help="Number of diffusion steps (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--guidance-scale", type=float, default=7.0,
                        help="CFG guidance scale (default: 7.0)")
    parser.add_argument("--caption", type=str,
                        default="pop, female vocal, piano, guitar, drums, bass, 120 bpm, C major, emotional",
                        help="Text caption prompt")
    parser.add_argument("--lyrics", type=str, default="",
                        help="Lyrics text (default: built-in test lyrics)")
    parser.add_argument("--bpm", type=int, default=120, help="BPM (default: 120)")
    parser.add_argument("--key", type=str, default="C major", help="Key/scale (default: C major)")
    parser.add_argument("--time-sig", type=str, default="4", help="Time signature (default: 4)")
    parser.add_argument("--vocal-lang", type=str, default="zh", help="Vocal language (default: zh)")
    parser.add_argument("--thinking", action="store_true", default=True,
                        help="Enable LM thinking (default: True)")
    parser.add_argument("--use-cot-metas", action="store_true", default=True,
                        help="Use CoT for metadata (default: True)")
    parser.add_argument("--use-cot-caption", action="store_true", default=True,
                        help="Use CoT for caption (default: True)")

    # Model / loading
    parser.add_argument("--model-root", type=str, default=str(_DEFAULT_MODEL_ROOT),
                        help="Model root directory (default: {})".format(_DEFAULT_MODEL_ROOT))
    parser.add_argument("--config", type=str, default="acestep-v15-sft",
                        help="Model config name (default: acestep-v15-sft)")
    parser.add_argument("--lm-model", type=str, default="acestep-5Hz-lm-1.7B",
                        help="LM model name (default: acestep-5Hz-lm-1.7B)")

    # Sinkhorn-RoPE parameters
    parser.add_argument("--use-sinkhorn-rope", action="store_true", default=False,
                        help="Enable Sinkhorn-Routed RoPE correction (deprecated: use --mode)")
    parser.add_argument("--rope-beta", type=float, default=1.0,
                        help="Position shift scaling factor (default: 1.0)")
    parser.add_argument("--max-token-shift", type=float, default=0.25,
                        help="Max token position shift in token units (default: 0.25, highest-freq phase ≤ 0.25 rad)")
    parser.add_argument("--no-full-attention-only", action="store_false", dest="full_attention_only", default=True,
                        help="Allow RoPE correction on sliding_attention layers too (default: only full_attention)")
    parser.add_argument("--rope-layers", type=str, default="12,16",
                        help="Comma-separated layer indices for RoPE correction (default: 12,16)")
    parser.add_argument("--rope-start-ratio", type=float, default=0.3,
                        help="Start ratio for RoPE correction in diffusion schedule (default: 0.3)")
    parser.add_argument("--rope-end-ratio", type=float, default=0.7,
                        help="End ratio for RoPE correction in diffusion schedule (default: 0.7)")
    parser.add_argument("--sinkhorn-iters", type=int, default=5,
                        help="Number of Sinkhorn iterations (default: 5)")
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.1,
                        help="Sinkhorn epsilon / temperature (default: 0.1)")
    parser.add_argument("--budget-smoothing-gamma", type=float, default=0.2,
                        help="Budget smoothing gamma (default: 0.2)")
    parser.add_argument("--output-dir", type=str, default="output/sinkhorn_routed_rope",
                        help="Output directory (default: output/sinkhorn_routed_rope)")

    args = parser.parse_args(argv)

    # Parse --rope-layers
    if args.rope_layers:
        try:
            args.rope_layers = [int(x.strip()) for x in args.rope_layers.split(",")]
        except ValueError:
            parser.error("--rope-layers must be comma-separated integers")
    else:
        args.rope_layers = []

    return args


# ---------------------------------------------------------------------------
# 8.  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
