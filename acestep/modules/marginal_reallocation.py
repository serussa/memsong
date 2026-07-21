"""
Sinkhorn-guided Marginal Reallocation for cross-attention.

Inference-only: computes a Sinkhorn coupling P between lyric lines, then
redistributes cross-attention mass across lines while preserving:
- Non-lyric token weights
- Total attention mass per lyric line
- Intra-line relative token distribution.

Usage:
    from acestep.modules.marginal_reallocation import apply_reallocation_to_model
    apply_reallocation_to_model(model, uth, t2u, lym, uil, lam=0.1)
"""

import math
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward, repeat_kv


# ================================================================
#  Log-domain Sinkhorn
# ================================================================

def log_sinkhorn(C, nu=None, mu=None, iters=10, epsilon=0.18):
    """
    C:      [U, U]  cost matrix (smaller = more similar)
    nu:     [U]     row marginal target (default uniform, each row→1)
    mu:     [U]     col marginal target (default uniform, each col→1)
    iters:  int     Sinkhorn iterations
    epsilon: float  entropic regularisation strength
    returns [U, U]  row-stochastic transport plan P
    """
    U = C.shape[-1]
    device = C.device
    if nu is None:
        nu = torch.ones(U, device=device)
    if mu is None:
        mu = torch.ones(U, device=device)
    # Normalise for numerical stability
    K = torch.exp(-C / epsilon)          # [U, U]
    a = torch.ones(U, device=device)
    b = torch.ones(U, device=device)
    for _ in range(iters):
        b = mu / (K.T @ a)
        a = nu / (K @ b)
    P = a[:, None] * K * b[None, :]      # [U, U]
    # Row-normalise for safety
    P = P / (P.sum(dim=-1, keepdim=True).clamp(min=1e-12))
    return P


# ================================================================
#  Coupling computation from line-level features
# ================================================================

def compute_line_coupling(uth, uil, sinkhorn_iters=10, sigma=0.18):
    """
    Compute P coupling between lyric lines.

    Args:
        uth:  [B, U, D]  unit-level text hidden states
        uil:  [U]         unit_is_lyric (bool)
        sinkhorn_iters, sigma: Sinkhorn params

    Returns:
        P:    [U, U]  row-stochastic coupling,  identity for non-lyric units
              0 ≤ P[u,v] ≤ 1,  sum_v P[u,v] = 1 (for lyric units)
    """
    U = uth.shape[1]
    device = uth.device
    uth_mean = uth.mean(dim=0)           # [U, D]

    P = torch.eye(U, device=device)
    lyric_units = uil.nonzero(as_tuple=True)[0]
    n_lyric = len(lyric_units)
    if n_lyric < 2:
        return P

    feats = uth_mean[lyric_units]        # [n_lyric, D]
    feats_n = F.normalize(feats, dim=-1)
    sim = feats_n @ feats_n.T             # [n_lyric, n_lyric]
    cost = 1.0 - sim.clamp(-1, 1)         # [n_lyric, n_lyric], [0, 2]

    P_sub = log_sinkhorn(cost, iters=sinkhorn_iters, epsilon=sigma)

    # Scatter back into full [U, U]
    P_lyric = torch.zeros(U, U, device=device)
    idx = lyric_units[:, None].expand(-1, n_lyric)
    jdx = lyric_units[None, :].expand(n_lyric, -1)
    P_lyric[idx, jdx] = P_sub
    # Non-lyric units remain identity on-diagonal, zero off-diagonal
    for u in range(U):
        if not uil[u]:
            P_lyric[u, u] = 1.0
    return P_lyric


# ================================================================
#  Marginal reallocation on attention weights
# ================================================================

def marginal_reallocation(attn, value, P, t2u, lyric_mask, lam=0.1,
                          return_weights=False, late_only_lambda_max=None,
                          lam_min=None):
    """
    Apply reallocation and return modified attention output (or weights).

    Args:
        attn:  [B, H, T, K]  post-softmax attention weights
        value: [B, H, K, D]  value states
        P:     [U, U]         row-stochastic coupling
        t2u:   [K]            token → unit (long)
        lyric_mask: [K]       bool (True = lyric token)
        lam:   float          soft-fusion weight (0 = no change)
        return_weights: bool  if True, return fused weights instead of output

    Returns:
        if return_weights: [B, H, T, K] fused attention weights
        else: [B, H, T, D] modified attention output
    """
    B, H, T, K = attn.shape
    U = P.shape[0]
    device = attn.device

    # Work in float32 for numerical stability
    attn_f = attn.float()
    P_f = P.float()
    lm_f = lyric_mask.to(device, dtype=torch.float32)
    t2u_d = t2u.to(device)

    if lam <= 0 and (late_only_lambda_max is None or late_only_lambda_max <= 0):
        return attn_f if return_weights else torch.matmul(attn_f, value.float())

    # ---- Position-dependent lambda ----
    # Priority: late-only > progressive (lam_min) > fixed
    if late_only_lambda_max is not None and late_only_lambda_max > 0:
        half_T = T // 2
        pos_lam = torch.zeros(T, device=device, dtype=torch.float32)
        if T > half_T:
            pos_lam[half_T:] = torch.linspace(0, late_only_lambda_max, T - half_T, device=device)
        pos_lam = pos_lam.view(1, 1, T, 1)
    elif lam_min is not None:
        # Progressive: λ(p) = lam_min + (lam - lam_min) * p  for p∈[0,1] across T
        p = torch.linspace(0, 1, T, device=device, dtype=torch.float32)
        pos_lam = (lam_min + max(lam - lam_min, 0.0) * p).view(1, 1, T, 1)
    else:
        pos_lam = lam

    # Unit membership matrix:  R_u[k] = 1 if token k belongs to unit u
    R = torch.zeros(U, K, device=device, dtype=torch.float32)
    R[t2u_d, torch.arange(K, device=device)] = 1.0
    R_lyric = R * lm_f  # [U, K]

    # ---- Step 1: aggregate attention mass per unit ----
    mass = torch.matmul(attn_f, R_lyric.T)   # [B, H, T, U]

    # ---- Step 2: redistribute ----
    new_mass = torch.matmul(mass, P_f.T)     # [B, H, T, U]

    # ---- Step 3: per-token scale factor ----
    mass_k = torch.zeros(B, H, T, K, device=device, dtype=torch.float32)
    new_mass_k = torch.zeros(B, H, T, K, device=device, dtype=torch.float32)
    for k in range(K):
        u = int(t2u_d[k])
        mass_k[..., k] = mass[..., u]
        new_mass_k[..., k] = new_mass[..., u]

    scale = torch.where(
        mass_k > 1e-12,
        (new_mass_k / mass_k).clamp(0.25, 4.0),
        torch.ones_like(mass_k)
    )

    # Apply only to lyric tokens
    mask = lm_f  # [K]
    scale = scale * mask[None, None, None, :] + (1 - mask[None, None, None, :])

    # ---- Step 4: soft fuse (using pos_lam: scalar or position-dependent) ----
    attn_modified = attn_f * scale
    attn_fused = (1 - pos_lam) * attn_f + pos_lam * attn_modified

    if return_weights:
        return attn_fused.to(dtype=attn.dtype)

    # ---- Step 5: recompute output ----
    return torch.matmul(attn_fused, value.float()).to(dtype=attn.dtype)


# ================================================================
#  Diagnostics
# ================================================================

def compute_diagnostics(attn_orig, attn_new, value, t2u, lyric_mask):
    """Collect per-layer diagnostics.

    Args:
        attn_orig: [B, H, T, K] original attention weights
        attn_new:  [B, H, T, K] modified attention weights (same shape as attn_orig)
        value:     [B, H, K, D]
    """
    B, H, T, K = attn_orig.shape
    device = attn_orig.device

    # Work in float32
    ao = attn_orig.float()
    an = attn_new.float()
    v = value.float() if value is not None else torch.zeros(1, device=device)
    t2u_d = t2u.to(device) if isinstance(t2u, torch.Tensor) else torch.tensor(t2u, device=device)

    # 1. Attention weight change ratio: ||A' - A|| / ||A||
    diff_norm = torch.norm(an - ao).item()
    orig_norm = torch.norm(ao).item()
    attn_change = diff_norm / (orig_norm + 1e-12)

    # 2. Context RMS change (over output, not weights)
    ctx_orig = torch.matmul(ao, v)
    ctx_new = torch.matmul(an, v)
    ctx_diff = torch.norm(ctx_new - ctx_orig).item()
    ctx_orig_n = torch.norm(ctx_orig).item()
    ctx_change = ctx_diff / (ctx_orig_n + 1e-12)

    # 3. Line-distribution error: mean χ² relative change in per-line attention mass
    U = t2u_d.max().item() + 1
    R = torch.zeros(U, K, device=device, dtype=torch.float32)
    R[t2u_d, torch.arange(K, device=device)] = 1.0
    mass_orig = torch.matmul(ao, R.T)   # [B, H, T, U]
    mass_new = torch.matmul(an, R.T)
    norm_o = mass_orig.sum(-1, keepdim=True).clamp(min=1e-12)
    norm_n = mass_new.sum(-1, keepdim=True).clamp(min=1e-12)
    line_err = ((mass_new / norm_n - mass_orig / norm_o) ** 2 /
                (mass_orig / norm_o + 1e-12)).mean().sqrt().item()

    return {
        'attn_change_ratio': round(attn_change, 6),
        'ctx_rms_change': round(ctx_change, 6),
        'line_distribution_error': round(line_err, 6),
    }


# ================================================================
#  Model patching
# ================================================================

def apply_reallocation_to_model(model, device, t2u, lym, uil, uth,
                                lam=0.1, layers=(8, 16),
                                sinkhorn_iters=10, sigma=0.18,
                                late_only_lambda_max=None,
                                step_gated=False, step_start=0.25, step_end=0.65,
                                num_steps=50,
                                lam_min=None):
    """
    Monkey-patch cross-attention of middle layers.

    Works by replacing the forward method of each middle layer's cross_attn
    module, forcing output_attentions=True so we get attention weights even
    when SDPA is the default backend.

    When step_gated=True, reallocation only activates for denoising steps
    where q = step/num_steps falls within [step_start, step_end]; outside
    this range λ=0 (no modification).

    Args:
        model: AceStepConditionGenerationModel
        device: torch device
        t2u:   [K] token_to_unit (long tensor on model device)
        lym:   [K] lyric_mask (bool tensor on model device)
        uil:   [U] unit_is_lyric (bool tensor on model device)
        uth:   [B, U, D] unit-level hidden states
        lam:   soft-fusion lambda (used when step gate is active)
        layers: (start, end) layer range (end exclusive)
        step_gated: enable denoising-step gating
        step_start: normalized step q where gate opens (default 0.25)
        step_end: normalized step q where gate closes (default 0.65)
        num_steps: total number of denoising steps
    """
    P = compute_line_coupling(uth, uil, sinkhorn_iters=sinkhorn_iters, sigma=sigma)
    P = P.to(device)
    print(f"[reallocation] P shape={P.shape}, lyric_units={uil.sum().item()}/{uil.shape[0]}"
          f", lam={lam}")

    # Patch eager_attention_forward in the actual runtime module.
    # We proved this works: replacing the module's eager_attention_forward
    # is seen by AceStepAttention.forward at runtime.
    import importlib
    _mod_name = ('transformers_modules.acestep_hyphen_v15_hyphen_sft.modeling_acestep_v15_base'
                 if 'sft' in str(model.__class__).lower() else
                 'transformers_modules.acestep_hyphen_v15_hyphen_base.modeling_acestep_v15_base')
    _attn_mod = importlib.import_module(_mod_name)

    # Ensure P dtype matches attention weights
    P = P.to(device=device, dtype=torch.bfloat16)

    rd = {
        'P': P.detach(),
        't2u': t2u.to(device) if t2u.device != device else t2u,
        'lyric_mask': lym.to(device) if lym.device != device else lym,
        'lam': lam,
        'late_only_lambda_max': late_only_lambda_max,
        'lam_min': lam_min,
        'step_gated': step_gated,
        'step_start': step_start,
        'step_end': step_end,
        'num_steps': num_steps,
        'current_step': -1,
        'diag_store': [],
        '_orig_forwards': [],
        '_decoder_hook': None,
    }

    # Register decoder pre-forward hook to track denoising step
    if step_gated:
        def _step_counter_hook(module, input):
            rd['current_step'] += 1
        rd['_decoder_hook'] = model.decoder.register_forward_pre_hook(_step_counter_hook)
        print(f"[reallocation] Step gate: q∈[{step_start},{step_end}] over {num_steps} steps, lam={lam}")

    # Force cross_attn layers to output attentions
    for layer_idx in range(layers[0], layers[1]):
        ca = model.decoder.layers[layer_idx].cross_attn
        orig_ca_forward = ca.forward
        def _force_output_attn(of):
            def _wrapped(*args, **kwargs):
                kwargs['output_attentions'] = True
                return of(*args, **kwargs)
            return _wrapped
        ca.forward = _force_output_attn(orig_ca_forward)
        rd['_orig_forwards'].append((ca, orig_ca_forward))

    # Guard against double-patching
    if not hasattr(_attn_mod, '_realloc_orig_eager'):
        _attn_mod._realloc_orig_eager = _attn_mod.eager_attention_forward

    _orig_eager = _attn_mod._realloc_orig_eager

    _mod = _attn_mod  # alias

    _called = [0]

    def _patched_eager(module, query, key, value, attention_mask,
                       scaling, dropout=0.0, **kwargs):
        _called[0] += 1
        res = _orig_eager(module, query, key, value, attention_mask,
                          scaling, dropout, **kwargs)
        if not getattr(module, 'is_cross_attention', False):
            return res
        attn_out, attn_w = res
        v_repeated = repeat_kv(value, module.num_key_value_groups)

        # ---- Step-gating: compute effective λ based on denoising step ----
        if rd.get('step_gated', False):
            cs = rd.get('current_step', 0)
            ns = rd.get('num_steps', 50)
            q = cs / ns if ns > 0 else 0.0
            s_start = rd.get('step_start', 0.25)
            s_end = rd.get('step_end', 0.65)
            inside = s_start <= q <= s_end
            lam_eff = rd.get('lam', 0.1) if inside else 0.0
        else:
            lam_eff = rd.get('lam', 0.1)
            q = -1.0

        if lam_eff <= 0 and rd.get('late_only_lambda_max') is None:
            # No reallocation this step — record empty diagnostic
            if rd['diag_store'] is not None:
                rd['diag_store'].append({
                    'step': rd.get('current_step', -1),
                    'q': q,
                    'effective_lam': 0.0,
                    'attn_change_ratio': 0.0,
                    'ctx_rms_change': 0.0,
                    'line_distribution_error': 0.0,
                })
            if _called[0] <= 2:
                print(f'[realloc] called #{_called[0]} step={rd.get("current_step",-1)} '
                      f'q={q:.3f} lam_eff=0 (outside gate)',
                      file=__import__('sys').stderr, flush=True)
            return res

        aw_mod = marginal_reallocation(attn_w, value, rd['P'], rd['t2u'],
                                       rd['lyric_mask'], lam=lam_eff,
                                       return_weights=True,
                                       late_only_lambda_max=rd.get('late_only_lambda_max'),
                                       lam_min=rd.get('lam_min'))
        new_out = torch.matmul(aw_mod, v_repeated.to(dtype=aw_mod.dtype))
        if rd['diag_store'] is not None:
            diag = compute_diagnostics(attn_w, aw_mod, v_repeated, rd['t2u'], rd['lyric_mask'])
            diag['step'] = rd.get('current_step', -1)
            diag['q'] = q
            diag['effective_lam'] = lam_eff
            rd['diag_store'].append(diag)
        if _called[0] <= 2:
            print(f'[realloc] called #{_called[0]} cross_attn={module.is_cross_attention} '
                  f'aw={list(attn_w.shape)} val={list(value.shape)}',
                  file=__import__('sys').stderr, flush=True)
        return new_out, attn_w

    _attn_mod.eager_attention_forward = _patched_eager

    print(f"[reallocation] Patched eager_attention_forward in {_mod_name}")
    return rd


def restore_eager_attention(rd=None):
    """Restore original eager_attention_forward and cross_attn forwards."""
    import importlib
    for mod_name in [
        'transformers_modules.acestep_hyphen_v15_hyphen_sft.modeling_acestep_v15_base',
        'transformers_modules.acestep_hyphen_v15_hyphen_base.modeling_acestep_v15_base',
    ]:
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, '_realloc_orig_eager'):
                m.eager_attention_forward = m._realloc_orig_eager
                del m._realloc_orig_eager
        except (ImportError, ModuleNotFoundError):
            pass

    # Restore per-layer cross_attn forwards
    if rd is not None:
        for ca, orig_fn in rd.get('_orig_forwards', []):
            ca.forward = orig_fn

    # Remove decoder pre-forward hook
    if rd is not None:
        hook = rd.get('_decoder_hook', None)
        if hook is not None:
            hook.remove()


def print_diagnostics(rd):
    """Print stored diagnostics. Accepts either rd dict or list of diag dicts."""
    diags = rd if isinstance(rd, list) else rd.get('diag_store', [])
    if not diags:
        print("[reallocation] No diagnostics collected")
        return

    has_step = 'step' in diags[0]
    if has_step:
        # Group by step
        from collections import defaultdict
        by_step = defaultdict(list)
        for d in diags:
            by_step[d['step']].append(d)
        print(f"[reallocation] Diagnostics by step ({len(diags)} total, {len(by_step)} steps):")
        print(f"  {'step':>4} {'q':>6} {'lam_eff':>8} {'attn_chg':>9} {'ctx_chg':>9} {'line_err':>9}")
        print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*9}")
        for step in sorted(by_step.keys()):
            dd = by_step[step]
            avg = {k: sum(d[k] for d in dd) / len(dd) for k in dd[0] if isinstance(dd[0][k], (int, float)) and k not in ('step',)}
            print(f"  {step:>4} {avg.get('q', -1):>6.3f} {avg.get('effective_lam', 0):>8.4f} "
                  f"{avg.get('attn_change_ratio', 0):>9.6f} {avg.get('ctx_rms_change', 0):>9.6f} "
                  f"{avg.get('line_distribution_error', 0):>9.6f}")

    # Overall average
    avg = {k: sum(d[k] for d in diags) / len(diags) for k in diags[0] if isinstance(diags[0][k], (int, float)) and k not in ('step', 'q')}
    print(f"\n[reallocation] Overall avg ({len(diags)} calls):")
    for k, v in avg.items():
        print(f"  {k}: {v:.6f}")
