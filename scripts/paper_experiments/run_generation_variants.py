#!/usr/bin/env python3
"""
Paper experiment: batch generation across model variants.

Runs inference for all configured model variants on a prompt set.
Each variant injects (or doesn't inject) a retrieval adapter at
layer 12 of the DiT backbone.

Variants
--------
  baseline         — No adapter (original ACE-Step).
  softmax_adapter  — Row-softmax over condition units (no Sinkhorn).
  transport_only   — Sinkhorn coupling, but transport_qk_scale=0
                     (no state-conditioned cost).
  full             — Full method: Sinkhorn + state-conditioned cost.

Optional:
  no_structural_units  — Only lyric units, no non-vocal structural units.
  no_state_cost        — Same as transport_only (qk_scale=0).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

# ---------------------------------------------------------------------------
# Imports that require the project environment
# ---------------------------------------------------------------------------
try:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    from acestep.phase_memory import (
        PMRetrievalPhaseMemory,
        TransportRetrievalAdapter,
        log_sinkhorn,
        build_duration_scaffold,
        parse_lyrics_to_units,
        scaffold_progress,
    )
    _IMPORT_OK = True
except ImportError as e:
    print(f"[WARN] Some imports failed: {e}", file=sys.stderr)
    print("[WARN] Running in dry-run / import-check mode.", file=sys.stderr)
    _IMPORT_OK = False


# ===========================================================================
#  RowSoftmaxRetrievalAdapter — softmax baseline adapter
# ===========================================================================

class RowSoftmaxRetrievalAdapter(torch.nn.Module):
    """Row-softmax retrieval adapter (no marginal constraints).

    Mirrors ``TransportRetrievalAdapter`` design but replaces Sinkhorn
    with row-softmax over condition units.  The transport plan is::

        A_ij = softmax_j(L_ij)
        Pi_ij = nu_i * A_ij

    where ``L = -C + qk_scale * R`` is the same combined logit as the
    full method.  No column marginal constraint is enforced.

    Purpose: ablation control — shows that adding *any* adapter is not
    enough; the marginal constraints in Sinkhorn are what matter.
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
        transport_sigma: float = 0.18,
        transport_qk_scale: float = 1.0,
        write_alpha_init: float = 1e-4,
        write_alpha_max: float = 1e-3,
        writer_eps: float = 1e-6,
        out_proj_init_std: float = 0.01,
    ):
        super().__init__()
        self.d_r = d_r
        self.time_dim = time_dim
        self.transport_sigma = transport_sigma
        self.transport_qk_scale = transport_qk_scale
        self.write_alpha_max = write_alpha_max
        self.writer_eps = writer_eps

        self.pm_state_norm = torch.nn.LayerNorm(pm_dim)

        self.audio_coord_mlp = torch.nn.Sequential(
            torch.nn.Linear(3, coord_dim), torch.nn.SiLU(),
            torch.nn.Linear(coord_dim, coord_dim), torch.nn.SiLU(),
        )
        self.unit_coord_mlp = torch.nn.Sequential(
            torch.nn.Linear(3, coord_dim), torch.nn.SiLU(),
            torch.nn.Linear(coord_dim, coord_dim), torch.nn.SiLU(),
        )
        self.section_embedding = torch.nn.Embedding(num_sections, coord_dim // 2)

        q_in_dim = pm_dim + coord_dim + time_dim
        self.q_mlp = torch.nn.Sequential(
            torch.nn.Linear(q_in_dim, d_r * 2), torch.nn.SiLU(),
            torch.nn.Linear(d_r * 2, d_r),
        )
        k_in_dim = text_dim + coord_dim + coord_dim // 2
        self.k_mlp = torch.nn.Sequential(
            torch.nn.Linear(k_in_dim, d_r * 2), torch.nn.SiLU(),
            torch.nn.Linear(d_r * 2, d_r),
        )
        self.v_mlp = torch.nn.Sequential(
            torch.nn.Linear(text_dim, d_r * 2), torch.nn.SiLU(),
            torch.nn.Linear(d_r * 2, d_r),
        )

        self.out_proj = torch.nn.Linear(d_r, hidden_dim)
        torch.nn.init.normal_(self.out_proj.weight, std=out_proj_init_std)
        torch.nn.init.zeros_(self.out_proj.bias)

        init_prob = max(write_alpha_init / max(write_alpha_max, 1e-10), 1e-6)
        init_prob = min(init_prob, 0.999)
        self.write_logit = torch.nn.Parameter(
            torch.tensor(np.log(init_prob / (1.0 - init_prob)))
        )
        self.silence_embedding = torch.nn.Parameter(
            torch.randn(1, 1, text_dim) * 0.02
        )
        self.null_value = torch.nn.Parameter(torch.zeros(1, 1, d_r))

    @property
    def write_alpha(self):
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
        unit_is_lyric: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        B, T_a = p_audio.shape
        K = unit_c_pos.shape[-1]
        device = p_audio.device
        dtype = p_audio.dtype

        if K == 0:
            return torch.zeros_like(hidden_states), torch.zeros(B, T_a, 0, device=device), {}

        # Silence embedding
        if unit_is_lyric is not None:
            unit_text_hidden = torch.where(
                unit_is_lyric.unsqueeze(-1),
                unit_text_hidden,
                self.silence_embedding.expand(B, K, -1),
            )

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
        q = torch.nn.functional.normalize(self.q_mlp(q_in), dim=-1, p=2)

        u_feat = torch.stack([unit_c_pos, unit_c_pos ** 2, 1.0 - unit_c_pos], dim=-1)
        unit_coord = self.unit_coord_mlp(u_feat)
        sec_emb = self.section_embedding(unit_section_id.long()) if unit_section_id is not None \
            else torch.zeros(B, K, self.section_embedding.embedding_dim, device=device, dtype=dtype)

        k_in = torch.cat([unit_text_hidden, unit_coord, sec_emb], dim=-1)
        k = torch.nn.functional.normalize(self.k_mlp(k_in), dim=-1, p=2)
        v = self.v_mlp(unit_text_hidden)
        if unit_is_lyric is not None:
            v = torch.where(unit_is_lyric.unsqueeze(-1), v, self.null_value.expand(B, K, -1))

        # Combined logit
        dist = p_audio[:, :, None] - unit_c_pos[:, None, :]
        C = (dist / self.transport_sigma) ** 2
        base_logit = -C
        R = torch.matmul(q, k.transpose(-1, -2))
        L = base_logit + self.transport_qk_scale * R

        # Row-softmax (no Sinkhorn)
        A = torch.softmax(L, dim=-1)  # [B, T, K]
        nu = torch.full((B, T_a,), 1.0 / T_a, device=device, dtype=dtype)
        Pi = nu.unsqueeze(-1) * A  # [B, T, K]

        # Context
        ctx = torch.matmul(Pi, v) / nu.unsqueeze(-1).clamp(min=1e-10)
        raw_res = self.out_proj(ctx)
        raw_rms = torch.sqrt(raw_res.pow(2).mean(dim=-1, keepdim=True) + self.writer_eps)
        unit_res = raw_res / raw_rms
        h_rms = torch.sqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True)).detach()
        delta_h = self.write_alpha * h_rms * unit_res

        with torch.no_grad():
            col_mass = Pi.sum(dim=1)  # [B, K]
            diag = {
                "transport_max": Pi.max().item(),
                "transport_mean": Pi.mean().item(),
                "col_error": (col_mass - unit_mass).abs().mean().item(),
                "entropy": (-Pi * (Pi + 1e-10).log()).sum(dim=-1).mean().item(),
                "col_mass_min": col_mass.min().item(),
                "col_mass_max": col_mass.max().item(),
                "write_alpha": self.write_alpha.item(),
                "write_ratio": (delta_h.norm(dim=-1).mean() /
                                (hidden_states.norm(dim=-1).mean() + 1e-8)).item(),
                "R_std": R.std().item(),
                "base_logit_std": base_logit.std().item(),
            }
            if unit_is_lyric is not None:
                silence_mask_f = (~unit_is_lyric).float()
                diag["silence_mass_target"] = (unit_mass * silence_mask_f).sum(dim=-1).mean().item()
                diag["lyric_usage_total"] = (col_mass * unit_is_lyric.float()).sum(dim=-1).mean().item()
                diag["silence_usage_total"] = (col_mass * silence_mask_f).sum(dim=-1).mean().item()

        return delta_h, Pi, diag


# ===========================================================================
#  Adapter variant factory
# ===========================================================================

def build_adapter(
    variant: str,
    hidden_dim: int,
    text_dim: int,
    pm_dim: int,
    device: torch.device,
    checkpoint: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Build PM + adapter for the given variant.

    Parameters
    ----------
    variant : str
        One of ``baseline``, ``softmax_adapter``, ``transport_only``, ``full``,
        ``no_structural_units``.
    hidden_dim : int
        Model hidden dimension (D).
    text_dim : int
        Text encoder hidden dimension.
    pm_dim : int
        PhaseMemory output dimension (2 * mem_dim).
    device : torch.device
    checkpoint : dict or None
        If provided, contains state dicts under ``phase_memory`` and
        ``retrieval_adapter`` (or ``transport_adapter``).

    Returns
    -------
    pm : PMRetrievalPhaseMemory or None
    adapter : TransportRetrievalAdapter or RowSoftmaxRetrievalAdapter or None
    """
    pm = PMRetrievalPhaseMemory(
        dim=hidden_dim, mem_dim=128, hidden_dim=256,
        normalize_internal_state=True,
    ).to(device).float().eval()

    if variant == "baseline":
        # No adapter at all
        return pm, None

    if variant in ("softmax_adapter", "transport_only", "full", "no_structural_units"):
        if variant == "softmax_adapter":
            qk_scale = 1.0
            AdapterCls = RowSoftmaxRetrievalAdapter
        elif variant == "transport_only":
            qk_scale = 0.0
            AdapterCls = TransportRetrievalAdapter
        elif variant == "full":
            qk_scale = 1.0
            AdapterCls = TransportRetrievalAdapter
        elif variant == "no_structural_units":
            qk_scale = 1.0
            AdapterCls = TransportRetrievalAdapter

        adapt = AdapterCls(
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            pm_dim=pm_dim,
            d_r=kwargs.get("retrieval_adapter_dim", 256),
            transport_sigma=0.18,
            transport_qk_scale=qk_scale,
            write_alpha_init=kwargs.get("write_alpha_init", 0.001),
            write_alpha_max=kwargs.get("write_alpha_max", 0.01),
            out_proj_init_std=kwargs.get("out_proj_init_std", 0.01),
        ).to(device).float().eval()

        if checkpoint is not None:
            pm_sd = checkpoint.get("phase_memory", {})
            adapt_sd = checkpoint.get("retrieval_adapter", checkpoint.get("transport_adapter", {}))
            if pm_sd:
                try:
                    pm.load_state_dict(pm_sd, strict=False)
                except Exception as e:
                    print(f"  [WARN] PM state dict partial load: {e}")
            if adapt_sd:
                try:
                    adapt.load_state_dict(adapt_sd, strict=False)
                except Exception as e:
                    print(f"  [WARN] Adapter state dict partial load: {e}")

        return pm, adapt

    raise ValueError(f"Unknown variant: {variant}")


# ===========================================================================
#  Prompt loading
# ===========================================================================

def load_prompts(prompts_path: str, num_prompts: Optional[int] = None) -> List[Dict]:
    """Load prompts from jsonl or csv."""
    path = Path(prompts_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    prompts = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
    elif path.suffix == ".csv":
        import csv
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(dict(row))
    else:
        raise ValueError(f"Unsupported prompt format: {path.suffix}")

    if num_prompts is not None and num_prompts < len(prompts):
        prompts = prompts[:num_prompts]

    # Normalise fields
    for p in prompts:
        p.setdefault("prompt_id", p.get("id", f"prompt_{prompts.index(p)}"))
        p.setdefault("lyrics", p.get("lyrics", ""))
        p.setdefault("caption", p.get("caption", ""))
        p.setdefault("duration", p.get("duration", 180))
        p.setdefault("bpm", p.get("bpm", None))
        p.setdefault("key", p.get("key", None))

    return prompts


# ===========================================================================
#  Variant label helpers
# ===========================================================================

VARIANT_LABELS = {
    "baseline": "Baseline (no adapter)",
    "softmax_adapter": "Row-Softmax Adapter",
    "transport_only": "Sinkhorn (qk_scale=0)",
    "full": "Full Method",
    "no_structural_units": "No Structural Units",
    "no_state_cost": "No State Cost",
}


# ===========================================================================
#  Main generation pipeline
# ===========================================================================

def setup_model(
    model_root: str,
    config_path: str,
    device: str = "cuda",
):
    """Initialize ACE-Step handler and return the model."""
    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=model_root,
        config_path=config_path,
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not success:
        raise RuntimeError(f"Model init failed: {status}")
    return handler


def run_variant_generation(
    variant: str,
    prompts: List[Dict],
    output_dir: str,
    handler: Any,
    llm_handler: Any,
    model: Any,
    seeds: List[int] = (0,),
    inference_steps: int = 50,
    max_duration: int = 240,
    checkpoint_path: Optional[str] = None,
    save_diagnostics: bool = True,
) -> int:
    """Run generation for one variant over all prompts.

    Returns the number of successfully generated samples.
    """
    variant_dir = Path(output_dir) / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if provided
    ckpt = None
    if checkpoint_path and os.path.exists(checkpoint_path) and variant != "baseline":
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print(f"  Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"  [WARN] Checkpoint load failed ({e}), using zero-init adapters")

    # Build adapter
    hidden_dim = model.config.hidden_size if hasattr(model, "config") else 2048
    text_dim = hidden_dim
    pm_dim = 256
    device = model.device if hasattr(model, "device") else torch.device("cuda")

    pm, adapter = build_adapter(
        variant, hidden_dim, text_dim, pm_dim, device, checkpoint=ckpt,
        retrieval_adapter_dim=256,
    )

    if adapter is not None:
        model._paper_pm = pm
        model._paper_adapter = adapter
    else:
        model._paper_pm = None
        model._paper_adapter = None

    # Pre-compute scaffold for this variant (shared across seeds)
    # This is populated per-prompt in the hook for consistency
    _hook_scaffold = {}
    _hook_diag_data = {}

    def _capture_enc_hook(module, inputs, kwargs):
        eh = kwargs.get("encoder_hidden_states", None)
        if eh is not None:
            _hook_scaffold["encoder_hidden_states"] = eh.float()

    def _inject_hook(module, inputs, output):
        if model._paper_adapter is None:
            return output  # baseline: passthrough

        h = output[0].float()
        B, T = h.shape[:2]
        dev = h.device

        text_hidden = _hook_scaffold.get("encoder_hidden_states")
        if text_hidden is None or text_hidden.shape[0] != B:
            text_hidden = h

        # Build scaffold from stored prompt data
        prompt_data = _hook_scaffold.get("prompt_data", {})
        lyrics_text = prompt_data.get("lyrics", "")

        if lyrics_text:
            L_eff = text_hidden.shape[1]
            # Dummy section_ids (zeros) — during inference we don't have
            # per-token section labels; parse_lyrics_to_units uses section_ids
            # only for tag-awareness which still works with zeros.
            fake_section_ids = torch.zeros(max(L_eff, 1), dtype=torch.long, device=dev)

            units, _, debug = parse_lyrics_to_units(
                lyrics_text, fake_section_ids.cpu(),
                auto_transition_ratios={
                    "intro": 0.0, "outro": 0.0,
                    "chorus_to_verse": 0.0, "chorus_to_bridge": 0.0,
                    "bridge_to_chorus": 0.0,
                },
            )
            tcm = debug.get("tag_control_mask", None)
            L_eff = text_hidden.shape[1]
            sc = build_duration_scaffold(units, text_len=L_eff, tag_control_mask=tcm)

            # For `no_structural_units`: filter to only lyric units
            if variant == "no_structural_units":
                units = [u for u in units if not u.is_silence]

            sc_dev = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
            token_to_unit = sc_dev["token_to_unit"]
            lyric_mask = sc_dev["lyric_mask"]
            lyric_unit_mask = sc_dev.get("lyric_unit_mask")
            unit_boundaries = sc_dev["unit_boundaries"]
            unit_duration = sc_dev["unit_duration"]
            unit_section_ids_sc = sc_dev.get("unit_section_ids")

            if lyric_unit_mask is not None:
                lyric_uids = torch.where(lyric_unit_mask)[0]
                K = len(lyric_uids)
            else:
                K = 0

            U = len(unit_boundaries) - 1
            if variant == "no_structural_units":
                all_uids = lyric_uids
            else:
                all_uids = torch.arange(U, device=dev)

            U_eff = len(all_uids)
            if U_eff == 0:
                return output

            eh_f = text_hidden.float()
            c_all = ((unit_boundaries[:-1] + unit_boundaries[1:]) / 2)[all_uids]
            mu_all = (unit_duration[all_uids].clone())
            mu_all = mu_all / mu_all.sum()

            usid_all = unit_section_ids_sc[all_uids] if unit_section_ids_sc is not None \
                else torch.zeros(U_eff, dtype=torch.long, device=dev)

            unit_is_lyric_local = lyric_unit_mask[all_uids] if lyric_unit_mask is not None \
                else torch.ones(U_eff, dtype=torch.bool, device=dev)

            # Pool text hidden per unit
            unit_text_hidden_list = []
            for uid in all_uids.tolist():
                is_lyric_u = unit_is_lyric_local[uid].item()
                if is_lyric_u:
                    token_mask = (token_to_unit == uid) & lyric_mask
                else:
                    token_mask = token_to_unit == uid
                token_mask_b = token_mask.unsqueeze(0).expand(B, -1)
                if token_mask_b.any():
                    pooled = eh_f[token_mask_b].view(B, -1, eh_f.shape[-1]).mean(dim=1)
                else:
                    pooled = torch.zeros(B, eh_f.shape[-1], device=dev, dtype=torch.float32)
                unit_text_hidden_list.append(pooled)
            unit_text_hidden = torch.stack(unit_text_hidden_list, dim=1)

            c_unit_b = c_all.unsqueeze(0).expand(B, -1)
            mu_b = mu_all.unsqueeze(0).expand(B, -1)
            usid_b = usid_all.unsqueeze(0).expand(B, -1)
            unit_is_lyric_b = unit_is_lyric_local.unsqueeze(0).expand(B, -1)

            p_audio = torch.linspace(0, 1, T, device=dev, dtype=torch.float32)
            p_audio = p_audio.unsqueeze(0).expand(B, -1)

            t_emb = torch.zeros(B, 128, device=dev, dtype=torch.float32)

            with torch.no_grad():
                pm_state = model._paper_pm(h, None)
                delta_h, Pi, diag = model._paper_adapter(
                    hidden_states=h,
                    text_hidden=eh_f,
                    pm_state=pm_state,
                    p_audio=p_audio,
                    unit_text_hidden=unit_text_hidden,
                    unit_c_pos=c_unit_b,
                    unit_mass=mu_b,
                    unit_section_id=usid_b,
                    unit_is_lyric=unit_is_lyric_b,
                )
                h_new = h + delta_h

            # Store diagnostics for saving (use first batch element)
            if save_diagnostics:
                Pi_np = Pi.detach().cpu().numpy() if Pi is not None else None
                if Pi_np is not None and Pi_np.ndim == 3:
                    Pi_np = Pi_np[0]  # CFG doubles batch; take first
                _hook_diag_data["Pi"] = Pi_np
                _hook_diag_data["mu"] = mu_b[0].detach().cpu().numpy() if mu_b is not None else None
                _hook_diag_data["unit_section_ids"] = usid_b[0].detach().cpu().numpy() if usid_b is not None else None
                _hook_diag_data["unit_is_lyric"] = unit_is_lyric_b[0].detach().cpu().numpy() if unit_is_lyric_b is not None else None
                _hook_diag_data["p_audio"] = p_audio[0].detach().cpu().numpy()
                _hook_diag_data["c_unit"] = c_unit_b[0].detach().cpu().numpy() if c_unit_b is not None else None
                _hook_diag_data["adapter_diag"] = diag

            return (h_new.to(dtype=output[0].dtype), *output[1:])

        return output

    # Register hooks
    pre_handle = model.decoder.register_forward_pre_hook(_capture_enc_hook, with_kwargs=True)
    hook_handle = model.decoder.layers[12].register_forward_hook(_inject_hook)

    count = 0
    try:
        for prompt in prompts:
            prompt_id = prompt["prompt_id"]
            lyrics = prompt.get("lyrics", "")
            caption = prompt.get("caption", "")
            duration = prompt.get("duration", 180)
            bpm = prompt.get("bpm", None)
            key_sig = prompt.get("key", None)

            # Store prompt for hook
            _hook_scaffold["prompt_data"] = prompt

            for seed in seeds:
                seed_dir = variant_dir / prompt_id / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                # Skip if already generated
                audio_path = seed_dir / "audio.wav"
                if audio_path.exists():
                    print(f"    [SKIP] {prompt_id}/seed_{seed} — already exists")
                    count += 1
                    continue

                print(f"    Generating {prompt_id}/seed_{seed}...")

                # Build generation params
                params = GenerationParams(
                    task_type="text2music",
                    caption=caption,
                    lyrics=lyrics,
                    instrumental=False,
                    bpm=bpm,
                    keyscale=key_sig,
                    timesignature="4",
                    vocal_language="zh",
                    duration=-1 if duration is None else duration,
                    inference_steps=inference_steps,
                    guidance_scale=7.0,
                    seed=seed,
                    thinking=True,
                    use_cot_metas=True,
                    use_cot_caption=True,
                    lm_temperature=0.75,
                )
                config = GenerationConfig(
                    batch_size=1,
                    audio_format="wav",
                    use_random_seed=False,
                )

                result = generate_music(
                    dit_handler=handler,
                    llm_handler=llm_handler,
                    params=params,
                    config=config,
                    save_dir=str(seed_dir),
                )

                if result.success:
                    # Generate audio.wav from the actual generated file
                    actual_audio_path = None
                    if hasattr(result, 'audios') and result.audios:
                        actual_audio_path = result.audios[0].get('path')
                    if actual_audio_path and os.path.exists(actual_audio_path):
                        # Rename generated file to audio.wav
                        import shutil
                        if os.path.abspath(actual_audio_path) != os.path.abspath(str(audio_path)):
                            shutil.move(actual_audio_path, str(audio_path))
                            # Also clean up any sidecar files (e.g. .flac, .json)
                            actual_stem = Path(actual_audio_path).stem
                            for sidecar in seed_dir.iterdir():
                                if sidecar.stem == actual_stem and sidecar.name != "audio.wav":
                                    sidecar.unlink(missing_ok=True)

                    # Save metadata
                    metadata = {
                        "prompt_id": prompt_id,
                        "variant": variant,
                        "seed": seed,
                        "lyrics": lyrics,
                        "caption": caption,
                        "duration": duration,
                        "bpm": bpm,
                        "key": key_sig,
                        "inference_steps": inference_steps,
                        "checkpoint": checkpoint_path or "none",
                        "adapter_type": type(model._paper_adapter).__name__
                        if model._paper_adapter is not None else "none",
                        "generated_audio_path": str(audio_path),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    with open(seed_dir / "metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)

                    with open(seed_dir / "generation_config.json", "w") as f:
                        json.dump({
                            "variant": variant,
                            "seed": seed,
                            "inference_steps": inference_steps,
                            "guidance_scale": 7.0,
                            "transport_qk_scale": getattr(adapter, "transport_qk_scale", None)
                            if adapter is not None else None,
                            "adapter_type": type(adapter).__name__ if adapter is not None else "none",
                        }, f, indent=2)

                    # Save diagnostics
                    if save_diagnostics and _hook_diag_data:
                        Pi_np = _hook_diag_data.get("Pi")
                        if Pi_np is not None:
                            np.savez(
                                seed_dir / "transport_diagnostics.npz",
                                Pi=Pi_np,
                                mu=_hook_diag_data.get("mu", np.array([])),
                                unit_section_ids=_hook_diag_data.get("unit_section_ids", np.array([])),
                                unit_is_lyric=_hook_diag_data.get("unit_is_lyric", np.array([])),
                                p_audio=_hook_diag_data.get("p_audio", np.array([])),
                                c_unit=_hook_diag_data.get("c_unit", np.array([])),
                            )

                    count += 1
                    print(f"    ✓ {prompt_id}/seed_{seed}")
                else:
                    print(f"    ✗ {prompt_id}/seed_{seed} failed: {result.error}")

    finally:
        pre_handle.remove()
        hook_handle.remove()

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Paper experiment: batch generation across model variants."
    )
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to prompts jsonl/csv")
    parser.add_argument("--output_dir", type=str, default="outputs/paper_eval",
                        help="Output directory root")
    parser.add_argument("--variants", type=str, default="baseline,softmax_adapter,transport_only,full",
                        help="Comma-separated variant list")
    parser.add_argument("--seeds", type=str, default="0,1",
                        help="Comma-separated seed list")
    parser.add_argument("--num_prompts", type=int, default=20,
                        help="Number of prompts to use")
    parser.add_argument("--max_duration", type=int, default=240,
                        help="Max audio duration in seconds")
    parser.add_argument("--inference_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to PM-retrieval checkpoint")
    parser.add_argument("--config", type=str, default="acestep-v15-sft",
                        help="Model config name")
    parser.add_argument("--model_root", type=str,
                        default="/root/autodl-tmp/Ace-Step1.5",
                        help="Model root directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print config without running")
    args = parser.parse_args()

    variant_list = [v.strip() for v in args.variants.split(",")]
    seed_list = [int(s) for s in args.seeds.split(",")]
    valid_variants = {"baseline", "softmax_adapter", "transport_only", "full",
                      "no_structural_units", "no_state_cost"}

    for v in variant_list:
        if v not in valid_variants:
            print(f"[ERROR] Unknown variant: {v}")
            sys.exit(1)

    prompts = load_prompts(args.prompts, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")

    if args.dry_run:
        print(f"\nDry-run config:")
        print(f"  output_dir: {args.output_dir}")
        print(f"  variants: {variant_list}")
        print(f"  seeds: {seed_list}")
        print(f"  num_prompts: {len(prompts)}")
        print(f"  inference_steps: {args.inference_steps}")
        print(f"  checkpoint: {args.checkpoint_path or 'none'}")
        total = len(prompts) * len(variant_list) * len(seed_list)
        print(f"  total generations: {total}")
        return

    if not _IMPORT_OK:
        print("[FATAL] Required imports not available. Check environment.")
        sys.exit(1)

    print("=" * 60)
    print("Paper Experiment: Generation Variants")
    print("=" * 60)

    # Initialize model once (shared across variants)
    print("\n[1] Initializing ACE-Step model...")
    handler = setup_model(args.model_root, args.config)
    model = handler.model.eval()
    print("  ✓ Model loaded")

    print("\n[2] Initializing LLM handler...")
    llm_handler = LLMHandler()
    llm_success = llm_handler.initialize(
        checkpoint_dir=args.model_root,
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )
    if not llm_success:
        print("  ✗ LLM init failed, trying non-LLM generation")
        llm_handler = None

    # Run each variant
    total_generated = 0
    for variant in variant_list:
        print(f"\n{'=' * 60}")
        print(f"[3] Variant: {variant} ({VARIANT_LABELS.get(variant, variant)})")
        print(f"{'=' * 60}")
        n = run_variant_generation(
            variant=variant,
            prompts=prompts,
            output_dir=args.output_dir,
            handler=handler,
            llm_handler=llm_handler,
            model=model,
            seeds=seed_list,
            inference_steps=args.inference_steps,
            max_duration=args.max_duration,
            checkpoint_path=args.checkpoint_path,
            save_diagnostics=True,
        )
        total_generated += n
        print(f"  → {n} samples generated for {variant}")

    print(f"\n{'=' * 60}")
    print(f"Done. Total samples: {total_generated}")
    print(f"Output: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
