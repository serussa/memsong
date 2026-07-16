"""
Transported Structural Memory (TSM) for ACE-Step 1.5.

Leverages the existing Sinkhorn coupling (P ∈ R^{B×T×U}) to:

  1. Pool audio hidden states into structural slots:  M = W_pool^T @ V
  2. Mix structural slots via bidirectional MHSA + FFN
  3. Broadcast mixed slots back to audio:  G = W_broadcast @ M_mixed
  4. Project to backbone dim and add as residual:  H' = H + W_o @ G

Full path:
    H → P^T H → Structural Slots → Slot Mixing → P H → H'

Supports four modes:
  - sinkhorn_only:      Pass-through (no TSM)
  - sinkhorn_pool_broadcast:  Pool → Broadcast only (no slot mixer)
  - sinkhorn_tsm:       Full Pool → Slot Mixer → Broadcast
  - softmax_tsm:        Use row-softmax cross-attention map instead of Sinkhorn P
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
#  TSMConfig
# ===========================================================================


@dataclass
class TSMConfig:
    """Configuration for Transported Structural Memory."""

    use_tsm: bool = False
    """Enable TSM module."""

    tsm_mode: str = "sinkhorn_only"
    """TSM mode: 'sinkhorn_only', 'sinkhorn_pool_broadcast', 'sinkhorn_tsm', 'softmax_tsm'."""

    tsm_layers: List[int] = field(default_factory=lambda: [12])
    """Layer indices where TSM is applied (1-indexed by convention)."""

    tsm_memory_dim: int = 256
    """Dimension of the structural slot memory."""

    tsm_num_heads: int = 4
    """Number of attention heads in the slot mixer."""

    tsm_ffn_dim: int = 512
    """Hidden dimension of the slot mixer FFN."""

    tsm_slot_layers: int = 1
    """Number of slot transformer layers (default: 1)."""

    tsm_dropout: float = 0.0
    """Dropout rate in the slot mixer."""

    tsm_detach_coupling: bool = True
    """If True, detach coupling before TSM forward (freeze Sinkhorn branch)."""

    tsm_zero_init_output: bool = True
    """If True, zero-initialise output_proj so H' == H at init."""

    tsm_epsilon: float = 1e-6
    """Small constant for numerical stability in mass normalisation."""


# ===========================================================================
#  SlotTransformerLayer — bidirectional MHSA + FFN for structural slots
# ===========================================================================


class SlotTransformerLayer(nn.Module):
    """A single bidirectional transformer layer for structural slots.

    No causal mask — slots attend to all other slots bidirectionally.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.attn_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, U, D] structural slot states.
            key_padding_mask: [B, U] bool, True = valid slot, False = padding.

        Returns:
            x': [B, U, D] updated slot states.
        """
        B, U, D = x.shape

        # ---- Self-attention (bidirectional, no causal mask) -----------------
        x_norm = self.attn_norm(x)
        q = self.q_proj(x_norm).view(B, U, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, U, Dh]
        k = self.k_proj(x_norm).view(B, U, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, U, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / scale  # [B, H, U, U]

        # Build attention mask from key_padding_mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, U], True = valid
            # We need [B, 1, 1, U] with -inf for invalid
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, U]
            attn_mask = attn_mask.expand(-1, self.num_heads, U, -1)  # [B, H, U, U]
            attn_logits = attn_logits.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights.to(v.dtype), v)  # [B, H, U, Dh]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, U, D)
        attn_out = self.o_proj(attn_out)

        x = x + attn_out

        # ---- FFN ------------------------------------------------------------
        x = x + self.ffn(self.ffn_norm(x))

        return x


# ===========================================================================
#  TransportedStructuralMemory
# ===========================================================================


class TransportedStructuralMemory(nn.Module):
    """Transported Structural Memory — structural slot bottleneck via Sinkhorn coupling.

    Design:
        H [B, T, D]  +  P [B, T, U]
        → column-normalise P → pool_weights [B, T, U]
        → V = RMSNorm(H) @ W_v  [B, T, Dm]
        → M = pool_weights^T @ V  [B, U, Dm]
        → M' = SlotTransformer(M)  [B, U, Dm]
        → row-normalise P → broadcast_weights [B, T, U]
        → G = broadcast_weights @ M'  [B, T, Dm]
        → O = W_o @ G  [B, T, D]
        → H' = H + O
    """

    def __init__(
        self,
        model_dim: int,
        memory_dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 512,
        slot_layers: int = 1,
        dropout: float = 0.0,
        epsilon: float = 1e-6,
        detach_coupling: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.epsilon = epsilon
        self.detach_coupling = detach_coupling

        # ---- Input projection: H → V (with RMSNorm) -------------------------
        self.input_norm = nn.LayerNorm(model_dim)
        self.value_proj = nn.Linear(model_dim, memory_dim)

        # ---- Slot transformer -----------------------------------------------
        self.slot_layers = nn.ModuleList([
            SlotTransformerLayer(
                dim=memory_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(slot_layers)
        ])

        # ---- Output projection: G → H (ZERO-INITIALISED) --------------------
        self.output_proj = nn.Linear(memory_dim, model_dim)
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        coupling: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
        detach_coupling: bool = True,
        enable_slot_mixer: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden_states: [B, T, D] audio hidden states.
            coupling: [B, T, U] Sinkhorn transport plan (or softmax map).
            condition_mask: [B, U] bool, True = valid structural unit.
            detach_coupling: If True, detach coupling to block gradients to Sinkhorn.
            enable_slot_mixer: If False, skip slot transformer (pool/broadcast only).

        Returns:
            output: [B, T, D] TSM residual to add to hidden_states.
            diagnostics: dict of per-forward diagnostics.
        """
        B, T, D = hidden_states.shape
        assert coupling.shape[0] == B and coupling.shape[1] == T, \
            f"coupling shape {coupling.shape} mismatch with hidden_states [{B}, {T}, {D}]"
        U = coupling.shape[2]
        device = hidden_states.device
        compute_dtype = hidden_states.dtype

        diagnostics: Dict[str, Any] = {}

        # ---- 0. Detach coupling if requested ----------------------------------
        if detach_coupling:
            p = coupling.detach()
        else:
            p = coupling

        # ---- 0b. Clamp and validate coupling -----------------------------------
        # Work in FP32 for numerical stability
        p_f32 = p.float().clamp(min=0.0, max=1.0)

        # ---- 0c. Apply condition mask ------------------------------------------
        if condition_mask is not None:
            # condition_mask: [B, U], True = valid
            c_mask_f32 = condition_mask.float().unsqueeze(1)  # [B, 1, U]
            p_f32 = p_f32 * c_mask_f32
            valid_slot_count = condition_mask.sum(dim=-1).float()  # [B]
        else:
            valid_slot_count = torch.full((B,), U, device=device, dtype=torch.float32)

        # ---- 1. Column-normalise P → pool_weights [B, T, U] --------------------
        # W^{pool}_{iu} = P_{iu} / (sum_j P_{ju} + eps)
        col_mass = p_f32.sum(dim=1, keepdim=True)  # [B, 1, U]
        col_mass = col_mass.clamp_min(self.epsilon)
        pool_weights = p_f32 / col_mass  # [B, T, U]

        # ---- 2. Project H → V --------------------------------------------------
        h_norm = self.input_norm(hidden_states)
        V = self.value_proj(h_norm)  # [B, T, Dm]

        # ---- 3. Pool: audio → structural slots ---------------------------------
        # M = pool_weights^T @ V  →  [B, U, Dm]
        M = torch.bmm(pool_weights.transpose(1, 2).to(V.dtype), V)  # [B, U, Dm]

        # ---- 3a. Mu-weighted centering: remove common hidden direction ----------
        # slot_mean = sum_u mu_u * M_u  where mu_u = col_mass_u / sum_v col_mass_v
        # This subtracts the coupling-mass-weighted average slot, forcing each slot
        # to represent its deviation from the common direction.  No new parameters.
        mu = col_mass / col_mass.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)  # [B, 1, U]
        slot_mean = torch.bmm(mu.to(compute_dtype), M)  # [B, 1, Dm]
        M_centered = M - slot_mean  # [B, U, Dm]

        # ---- Slot diagnostics (no-grad) -----------------------------------------
        with torch.no_grad():
            # -- Raw (un-centered) slot RMS --
            slot_rms = torch.sqrt(M.pow(2).mean(dim=-1))  # [B, U]
            diagnostics["tsm_slot_rms"] = slot_rms.mean().item()
            diagnostics["tsm_slot_rms_min"] = slot_rms.min().item()
            diagnostics["tsm_slot_rms_max"] = slot_rms.max().item()

            # -- Coupling column-profile cosine --
            p_col_normed = F.normalize(pool_weights.float(), dim=1, p=2)
            p_col_cos = torch.bmm(p_col_normed.transpose(1, 2), p_col_normed)  # [B, U, U]
            if U >= 2:
                triu_idx = torch.triu_indices(U, U, offset=1, device=device)
                col_cos_vals = p_col_cos[:, triu_idx[0], triu_idx[1]]
                diagnostics["coupling_column_cosine_mean"] = col_cos_vals.mean().item()
            else:
                diagnostics["coupling_column_cosine_mean"] = 0.0

            # -- Raw slot cosine (un-centered M) --
            M_norm = F.normalize(M.float(), dim=-1, p=2)
            raw_cos = torch.bmm(M_norm, M_norm.transpose(1, 2))
            if U >= 2:
                triu_idx = torch.triu_indices(U, U, offset=1, device=device)
                raw_triu = raw_cos[:, triu_idx[0], triu_idx[1]]
                diagnostics["tsm_raw_slot_cosine_mean"] = raw_triu.mean().item()
            else:
                diagnostics["tsm_raw_slot_cosine_mean"] = 0.0

            # -- Mu-centered slot cosine (actual mixer input) --
            Mc_norm = F.normalize(M_centered.float(), dim=-1, p=2)
            center_cos = torch.bmm(Mc_norm, Mc_norm.transpose(1, 2))
            if U >= 2:
                triu_idx = torch.triu_indices(U, U, offset=1, device=device)
                center_triu = center_cos[:, triu_idx[0], triu_idx[1]]
                diagnostics["tsm_centered_slot_cosine_mean"] = center_triu.mean().item()
            else:
                diagnostics["tsm_centered_slot_cosine_mean"] = 0.0

            # -- Centered-to-raw RMS ratio (how much energy is in the common dir) --
            raw_rms = torch.sqrt(M.float().pow(2).mean(dim=-1))  # [B, U]
            cent_rms = torch.sqrt(M_centered.float().pow(2).mean(dim=-1))  # [B, U]
            rms_ratio = cent_rms / raw_rms.clamp_min(1e-10)
            diagnostics["tsm_centered_to_raw_rms_ratio"] = rms_ratio.mean().item()

            # -- Centered slot effective rank (P_eff from SVD) --
            # If slot collapse → low rank; well-differentiated → high rank
            try:
                _, S, _ = torch.linalg.svd(M_centered.float(), full_matrices=False)  # [B, min(U, Dm)]
                S_sum = S.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                S_norm = S / S_sum
                entropy = -(S_norm * torch.log(S_norm.clamp_min(1e-10))).sum(dim=-1)
                diagnostics["tsm_centered_slot_effective_rank"] = entropy.exp().mean().item()
            except RuntimeError:
                diagnostics["tsm_centered_slot_effective_rank"] = -1.0

        # ---- 4. Slot mixing (optional) ------------------------------------------
        if enable_slot_mixer and len(self.slot_layers) > 0:
            # Build key_padding_mask for slot transformer
            if condition_mask is not None:
                key_padding_mask = condition_mask  # [B, U], True = valid
            else:
                key_padding_mask = None

            M_mixed = M_centered.to(dtype=compute_dtype)
            for layer in self.slot_layers:
                M_mixed = layer(M_mixed, key_padding_mask=key_padding_mask)

            # Zero out padding slots after mixing
            if condition_mask is not None:
                pad_mask_expanded = condition_mask.unsqueeze(-1).to(M_mixed.dtype)  # [B, U, 1]
                M_mixed = M_mixed * pad_mask_expanded
        else:
            M_mixed = M_centered

        # ---- 4a. Re-center mixed slots (same mu, prevents re-emergence of common dir)
        M_mixed_centered = M_mixed - torch.bmm(mu.to(M_mixed.dtype), M_mixed)

        # ---- Mixed slot diagnostics ---------------------------------------------
        with torch.no_grad():
            mixed_rms = torch.sqrt(M_mixed.pow(2).mean(dim=-1))  # [B, U]
            diagnostics["tsm_mixed_slot_rms"] = mixed_rms.mean().item()

            # Slot cosine similarity (collapse check)
            if U >= 2:
                M_norm = F.normalize(M_mixed.float(), dim=-1, p=2)  # [B, U, Dm]
                slot_cos = torch.bmm(M_norm, M_norm.transpose(1, 2))  # [B, U, U]
                # Upper triangle, excluding diagonal
                if U > 1:
                    triu_idx = torch.triu_indices(U, U, offset=1, device=device)
                    triu_vals = slot_cos[:, triu_idx[0], triu_idx[1]]  # [B, N_pairs]
                    diagnostics["tsm_mixed_slot_cosine_mean"] = triu_vals.mean().item()
                    diagnostics["tsm_mixed_slot_cosine_max"] = triu_vals.max().item()
                else:
                    diagnostics["tsm_mixed_slot_cosine_mean"] = 0.0
                    diagnostics["tsm_mixed_slot_cosine_max"] = 0.0
            else:
                diagnostics["tsm_mixed_slot_cosine_mean"] = 0.0
                diagnostics["tsm_mixed_slot_cosine_max"] = 0.0

            # -- Centered mixed slot cosine (after re-centering, what gets broadcast)
            if U >= 2:
                Mc_norm = F.normalize(M_mixed_centered.float(), dim=-1, p=2)
                mc_cos = torch.bmm(Mc_norm, Mc_norm.transpose(1, 2))
                if U > 1:
                    triu_idx = torch.triu_indices(U, U, offset=1, device=device)
                    mc_triu = mc_cos[:, triu_idx[0], triu_idx[1]]
                    diagnostics["tsm_centered_mixed_cosine_mean"] = mc_triu.mean().item()
                else:
                    diagnostics["tsm_centered_mixed_cosine_mean"] = 0.0
            else:
                diagnostics["tsm_centered_mixed_cosine_mean"] = 0.0

        # ---- 5. Row-normalise P → broadcast_weights [B, T, U] -------------------
        # W^{broadcast}_{iu} = P_{iu} / (sum_v P_{iv} + eps)
        row_mass = p_f32.sum(dim=-1, keepdim=True)  # [B, T, 1]
        row_mass = row_mass.clamp_min(self.epsilon)
        broadcast_weights = p_f32 / row_mass  # [B, T, U]

        # ---- 6. Broadcast: structural slots → audio ----------------------------
        # G = broadcast_weights @ M_mixed_centered  →  [B, T, Dm]
        G = torch.bmm(broadcast_weights.to(M_mixed_centered.dtype), M_mixed_centered)  # [B, T, Dm]

        # ---- 7. Project to model dim (ZERO-INIT → O ≈ 0 at init) ---------------
        output = self.output_proj(G)  # [B, T, D]

        # ---- Diagnostics --------------------------------------------------------
        with torch.no_grad():
            hidden_rms = torch.sqrt(hidden_states.pow(2).mean(dim=-1))  # [B, T]
            output_rms = torch.sqrt(output.pow(2).mean(dim=-1))  # [B, T]

            diagnostics["tsm_hidden_rms"] = hidden_rms.mean().item()
            diagnostics["tsm_value_rms"] = torch.sqrt(V.pow(2).mean(dim=-1)).mean().item()
            diagnostics["tsm_output_rms"] = output_rms.mean().item()
            r_ratio = output_rms / (hidden_rms + self.epsilon)
            diagnostics["tsm_output_to_hidden_ratio"] = r_ratio.mean().item()
            diagnostics["tsm_output_to_hidden_ratio_max"] = r_ratio.max().item()

            # Column/row mass diagnostics
            c_mass = p_f32.sum(dim=1)  # [B, U]
            r_mass = p_f32.sum(dim=-1)  # [B, T]
            diagnostics["tsm_column_mass_min"] = c_mass.min().item()
            diagnostics["tsm_column_mass_max"] = c_mass.max().item()
            diagnostics["tsm_row_mass_min"] = r_mass.min().item()
            diagnostics["tsm_row_mass_max"] = r_mass.max().item()

            # Near-zero column count
            near_zero_cols = (c_mass < self.epsilon).float().sum(dim=-1)  # [B]
            diagnostics["tsm_near_zero_column_count"] = near_zero_cols.mean().item()
            diagnostics["tsm_near_zero_column_rate"] = (near_zero_cols / max(U, 1)).mean().item()
            diagnostics["tsm_valid_slot_count"] = valid_slot_count.mean().item()

            # NaN/Inf check
            diagnostics["tsm_output_has_nan"] = float(not torch.isfinite(output).all())
            diagnostics["tsm_output_has_inf"] = float(torch.isinf(output).any())

        return output, diagnostics


# ===========================================================================
#  TSM softmax coupling builder
# ===========================================================================


def build_softmax_coupling(
    hidden_states: torch.Tensor,
    text_hidden: torch.Tensor,
    p_audio: torch.Tensor,
    unit_c_pos: torch.Tensor,
    unit_mass: torch.Tensor,
    unit_is_lyric: Optional[torch.Tensor] = None,
    head_dim: int = 64,
    num_heads: int = 4,
) -> torch.Tensor:
    """Build a row-softmax coupling map for softmax_tsm mode.

    This replaces the Sinkhorn coupling with a simple cross-attention softmax,
    for ablation to verify that marginal-constrained coupling matters.

    Args:
        hidden_states: [B, T, D] audio hidden states.
        text_hidden: [B, U, D] pooled text hidden per unit.
        p_audio: [B, T] audio progress in [0, 1].
        unit_c_pos: [B, U] unit centre positions in [0, 1].
        unit_mass: [B, U] unit duration mass.
        unit_is_lyric: [B, U] bool, True = lyric unit.
        head_dim: int, dimension per head for Q/K.
        num_heads: int, number of heads (averaged over).

    Returns:
        softmax_map: [B, T, U] row-normalised softmax attention map.
    """
    B, T, D = hidden_states.shape
    U = unit_c_pos.shape[-1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Simple linear Q/K projections
    q_proj = nn.Linear(D, num_heads * head_dim, device=device, dtype=dtype)
    k_proj = nn.Linear(D, num_heads * head_dim, device=device, dtype=dtype)

    q = q_proj(hidden_states).view(B, T, num_heads, head_dim)  # [B, T, H, Dh]
    k = k_proj(text_hidden).view(B, U, num_heads, head_dim)    # [B, U, H, Dh]

    # Average over heads for a single coupling map
    scale = math.sqrt(head_dim)
    attn_logits = torch.einsum("bthd,buhd->bhtu", q, k) / scale  # [B, H, T, U]
    attn_logits = attn_logits.mean(dim=1)  # [B, T, U]

    # Add position bias
    dist = p_audio[:, :, None] - unit_c_pos[:, None, :]
    pos_bias = -(dist / 0.18) ** 2  # same sigma as transport
    attn_logits = attn_logits + pos_bias

    # Mask invalid units
    if unit_is_lyric is not None:
        attn_logits = attn_logits.masked_fill(~unit_is_lyric.unsqueeze(1), float("-inf"))

    softmax_map = F.softmax(attn_logits, dim=-1, dtype=torch.float32)  # [B, T, U]
    softmax_map = torch.nan_to_num(softmax_map, nan=0.0)

    # Scale by row mass from unit_mass to approximate coupling scale
    # (row-softmax gives uniform mass per row, not marginal-constrained)
    nu = 1.0 / T
    softmax_map = nu * softmax_map

    return softmax_map


# ===========================================================================
#  Gradient norm helper
# ===========================================================================


def _grad_norm(module: nn.Module) -> float:
    """Compute L2 gradient norm of all parameters in *module*."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return math.sqrt(total)


def collect_tsm_grad_norms(tsm: TransportedStructuralMemory) -> Dict[str, float]:
    """Collect gradient norms of all TSM sub-modules."""
    return {
        "output_proj_grad_norm": _grad_norm(tsm.output_proj),
        "value_proj_grad_norm": _grad_norm(tsm.value_proj),
        "input_norm_grad_norm": _grad_norm(tsm.input_norm),
        "slot_attn_qkv_grad_norm": sum(_grad_norm(l.q_proj) + _grad_norm(l.k_proj) + _grad_norm(l.v_proj)
                                       for l in tsm.slot_layers),
        "slot_attn_o_grad_norm": sum(_grad_norm(l.o_proj) for l in tsm.slot_layers),
        "slot_ffn_grad_norm": sum(_grad_norm(l.ffn) for l in tsm.slot_layers),
    }


# ===========================================================================
#  Smoke test
# ===========================================================================


def smoke_test_tsm():
    """Run minimal verification of the TSM module."""
    print("=" * 60)
    print("TSM smoke test")
    print("=" * 60)

    B, T, U, D = 2, 128, 12, 2048
    Dm = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. Basic forward ----------------------------------------------------
    print("\n[1] Basic forward...")
    H = torch.randn(B, T, D, device=device)
    # Generate a plausible coupling (row-stochastic-ish)
    raw = torch.randn(B, T, U, device=device)
    P = F.softmax(raw, dim=-1)  # row-softmax
    # Make it also column-normalised-ish
    P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-6) * (1.0 / U)
    P = P / P.sum(dim=-1, keepdim=True).clamp(min=1e-6) * (1.0 / T)

    condition_mask = torch.ones(B, U, dtype=torch.bool, device=device)

    tsm = TransportedStructuralMemory(
        model_dim=D, memory_dim=Dm, num_heads=4, ffn_dim=512,
        slot_layers=1, dropout=0.0,
    ).to(device)

    output, diag = tsm(H, P, condition_mask=condition_mask)

    assert output.shape == (B, T, D), f"output shape {output.shape} != ({B}, {T}, {D})"
    print(f"  ✓ output shape: {output.shape}")
    print(f"  ✓ tsm_output_to_hidden_ratio: {diag['tsm_output_to_hidden_ratio']:.2e}")
    assert not diag["tsm_output_has_nan"], "output has NaN"
    assert not diag["tsm_output_has_inf"], "output has Inf"
    print("  ✓ no NaN / Inf")

    # ---- 2. Zero-init check --------------------------------------------------
    print("\n[2] Zero-init check...")
    # Reset output_proj to zero
    nn.init.zeros_(tsm.output_proj.weight)
    if tsm.output_proj.bias is not None:
        nn.init.zeros_(tsm.output_proj.bias)
    output_zero, _ = tsm(H, P, condition_mask=condition_mask)
    zero_rms = torch.sqrt(output_zero.pow(2).mean())
    assert zero_rms < 1e-8, f"zero-init output RMS {zero_rms.item():.2e} ≠ 0"
    print(f"  ✓ H' == H at zero init (RMS={zero_rms.item():.2e})")

    # ---- 3. Padding slots ----------------------------------------------------
    print("\n[3] Padding slots...")
    # Mask out last 4 units
    pad_mask = torch.ones(B, U, dtype=torch.bool, device=device)
    pad_mask[:, -4:] = False

    output_pad, diag_pad = tsm(H, P, condition_mask=pad_mask)
    # Padding slots should have zero contribution
    assert not diag_pad["tsm_output_has_nan"], "output has NaN with padding"
    print(f"  ✓ padding slots handled (near_zero_col_rate={diag_pad['tsm_near_zero_column_rate']:.2f})")

    # ---- 4. No slot mixer ----------------------------------------------------
    print("\n[4] Pool/Broadcast only (no slot mixer)...")
    output_pb, diag_pb = tsm(H, P, condition_mask=condition_mask, enable_slot_mixer=False)
    assert output_pb.shape == (B, T, D), f"pool/broadcast output shape {output_pb.shape}"
    print(f"  ✓ pool/broadcast shape: {output_pb.shape}")

    # ---- 5. Backward test ----------------------------------------------------
    print("\n[5] Backward test...")
    tsm2 = TransportedStructuralMemory(
        model_dim=D, memory_dim=Dm, num_heads=4, ffn_dim=512,
        slot_layers=1, dropout=0.0,
    ).to(device)
    H_grad = H.clone().requires_grad_(True)
    P_grad = P.clone()

    output_grad, _ = tsm2(H_grad, P_grad, condition_mask=condition_mask, detach_coupling=True)
    loss = output_grad.sum()
    loss.backward()

    has_grad = sum(1 for p in tsm2.parameters() if p.grad is not None)
    print(f"  ✓ {has_grad} parameter groups have gradients")

    # ---- 6. FP32 normalization check ------------------------------------------
    print("\n[6] FP32 normalization...")
    # Run with extreme values to check no NaN
    P_extreme = torch.zeros(B, T, U, device=device)
    P_extreme[:, :, 0] = 1.0  # single active column
    output_ext, diag_ext = tsm(H, P_extreme, condition_mask=condition_mask)
    assert not diag_ext["tsm_output_has_nan"], "output has NaN with extreme coupling"
    print(f"  ✓ extreme coupling handled (near_zero_col_rate={diag_ext['tsm_near_zero_column_rate']:.2f})")

    # ---- 7. bf16 forward ------------------------------------------------------
    if device == "cuda":
        print("\n[7] bf16 forward...")
        tsm_bf16 = tsm.to(torch.bfloat16)
        H_bf16 = H.to(torch.bfloat16)
        P_bf16 = P.to(torch.bfloat16)
        output_bf16, diag_bf16 = tsm_bf16(H_bf16, P_bf16, condition_mask=condition_mask)
        assert output_bf16.shape == (B, T, D), f"bf16 output shape {output_bf16.shape}"
        assert not torch.isnan(output_bf16).any(), "bf16 output has NaN"
        print(f"  ✓ bf16 forward successful")

    # ---- 8. Gradient flow through non-zero output_proj -----------------------
    print("\n[8] Gradient flow with non-zero init...")
    tsm3 = TransportedStructuralMemory(
        model_dim=D, memory_dim=Dm, num_heads=4, ffn_dim=512,
        slot_layers=1, dropout=0.0,
    ).to(device)
    # Use small normal init instead of zero
    nn.init.normal_(tsm3.output_proj.weight, std=1e-4)

    H_grad2 = H.clone().requires_grad_(True)
    output_grad2, _ = tsm3(H_grad2, P, condition_mask=condition_mask)
    loss2 = output_grad2.sum()
    loss2.backward()

    for name, param in tsm3.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            if g_norm > 0:
                print(f"  {name}: grad_norm={g_norm:.6e}")

    print("\n" + "=" * 60)
    print("ALL TSM smoke tests passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    smoke_test_tsm()
