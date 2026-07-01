#!/usr/bin/env python3
"""
Section-RoPE Cross-Attention Diagnostic.

Captures layer 12 cross-attention maps under four conditions and compares
attention structure to verify Section-RoPE is doing something meaningful.

Conditions:
  baseline  — no Section-RoPE offset applied
  real      — trained Section-RoPE with correct section_ids
  unknown   — all section_ids = 0 (UNKNOWN, offset forced to 0)
  shuffled  — randomly permuted section_ids

Metrics per condition:
  1. Per-section attention fraction
  2. Attention entropy (lower = more focused)
  3. Section-conditioned attention variance (higher = more differentiation)
  4. Step-to-step pattern stability
"""

import os
import sys
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import safetensors.torch

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
SECTION_ROPE_DIR = Path("/root/autodl-tmp/section_ckpt/final")
OUTPUT_DIR = ACE_STEP_ROOT / "output"

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.tgca.section_rope import SectionRoPEOffset
from acestep.tgca.lyrics_parser import LyricsStructureParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTION_NAMES = {0: "UNKNOWN", 1: "INTRO", 2: "VERSE", 3: "PRECHORUS",
                 4: "CHORUS", 5: "BRIDGE", 6: "OUTRO", 7: "INSTR"}

LYRICS = """[Verse]
爱总忽然退潮 心慌乱触礁 沉没在深海里 看海面闪耀
但回忆像水草 紧紧的缠绕 梦才温热眼角 就冰冷掉
努力越过风暴 向着未来飘 我们才会遇到 感动的拥抱
你总是能知道 我的坚强剩多少
[Pre-chorus]
给我最刚好的依靠
[Chorus]
你手心的太阳 只轻放在我背上 委屈就能笑着落泪 被释放
在手心的太阳 黑暗里特别明亮 让远路好像 是一种分享 而不是漫长
[Inst]
让眼睛看不到 嫉妒的燃烧
[Verse]
让耳朵听不到 谎言的吵闹 再没有人相信 爱能永恒那一秒 我们正坚定的微笑
[Chorus]
你手心的太阳 有种安定的力量 就算世界再乱我也 不心慌
我手心的太阳 或许只像个月亮 却用所有爱 为你投射我最暖的光芒
你手心的太阳 只轻放在我背上
[Bridge]
委屈就能笑着落泪 被释放 在手心的太阳 黑暗里特别明亮
让远路好像 是一种分享 你手心的太阳 有种安定的力量
[Chorus]
就算世界再乱我也 不心慌 我手心的太阳 或许只像个月亮
却用所有爱 为你投射我 最暖的光芒"""

# ---------------------------------------------------------------------------
# Attention capture helper
# ---------------------------------------------------------------------------


class AttentionCapture:
    """Context manager: force output_attentions=True on layer12 cross-attn
    and capture every forward call."""

    def __init__(self, model):
        self.attentions: list[torch.Tensor] = []
        self._attn_module = model.decoder.layers[12].cross_attn
        self._orig_forward = None
        self._hook_handle = None

    def _hook(self, module, _input, output):
        if output[1] is not None:
            self.attentions.append(output[1].detach().cpu())

    def __enter__(self):
        self._orig_forward = self._attn_module.forward
        def _patched(*args, **kwargs):
            kwargs["output_attentions"] = True
            return self._orig_forward(*args, **kwargs)
        self._attn_module.forward = _patched
        self._hook_handle = self._attn_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self._hook_handle is not None:
            self._hook_handle.remove()
        if self._orig_forward is not None:
            self._attn_module.forward = self._orig_forward

    @property
    def tensor(self) -> torch.Tensor | None:
        if not self.attentions:
            return None
        return torch.stack(self.attentions, dim=0)  # [N, B, H, T, L]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_attention_metrics(attn: torch.Tensor, section_ids: torch.Tensor) -> dict:
    """Return dict of scalar metrics.

    Args:
        attn: [N, B, H, T, L]  — N = decoder forward calls across diffusion steps
        section_ids: [B, L]  — integer section type IDs
    """
    eps = 1e-8

    # [N, B, H, T, L] -> [H, T, L]  (avg over steps and batch)
    avg = attn.float().mean(dim=(0, 1))
    H, T, L = avg.shape

    unique_types = sorted(section_ids.unique().tolist())

    # ---- 1. Per-section attention fraction ----
    sid_mask = section_ids[0]  # [L]
    section_attn = {}
    for sid in unique_types:
        mask = (sid_mask == sid).float().view(1, 1, L)  # [1, 1, L]
        section_attn[sid] = (avg * mask).sum(dim=-1)  # [H, T]

    total_attn = sum(section_attn.values()) + eps
    section_frac = {sid: (v / total_attn).mean().item()
                    for sid, v in section_attn.items()}

    # ---- 2. Attention entropy ----
    p = avg / (avg.sum(dim=-1, keepdim=True) + eps)
    entropy = -(p * torch.log(p + eps)).sum(dim=-1)  # [H, T]
    avg_entropy = entropy.mean().item()
    max_entropy = math.log(L)
    norm_entropy = avg_entropy / max_entropy if max_entropy > 0 else 1.0

    # ---- 3. Section attention variance ----
    frac_stack = torch.stack(list(section_attn.values()), dim=-1)  # [H, T, n_types]
    frac_stack = frac_stack / (frac_stack.sum(dim=-1, keepdim=True) + eps)
    section_var = frac_stack.var(dim=-1).mean().item()

    # ---- 4. Step-to-step stability ----
    N = attn.shape[0]
    if N >= 2:
        correlations = []
        for i in range(N - 1):
            a = attn[i].float()  # [B, H, T, L]
            b = attn[i + 1].float()
            a_p = a / (a.sum(dim=-1, keepdim=True) + eps)
            b_p = b / (b.sum(dim=-1, keepdim=True) + eps)
            corr = torch.nn.functional.cosine_similarity(
                a_p.flatten(1), b_p.flatten(1), dim=1
            )
            correlations.append(corr.mean().item())
        stability = float(np.mean(correlations))
    else:
        stability = 0.0

    return {
        "section_frac": {SECTION_NAMES.get(s, str(s)): f
                         for s, f in section_frac.items()},
        "entropy": round(avg_entropy, 4),
        "norm_entropy": round(norm_entropy, 4),
        "section_var": round(section_var, 6),
        "stability": round(stability, 4),
        "num_decoder_calls": N,
    }


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------


def inject_section_rope(model, dtype, device):
    """Inject SectionRoPEOffset on layer 12 if not already present."""
    layer12 = model.decoder.layers[12]
    if getattr(layer12, "use_section_rope", False):
        return
    model.config.use_section_rope_offset = True
    model.config.use_token_weights = True
    layer12.use_section_rope = True
    layer12.section_rope_offset_module = (
        SectionRoPEOffset(num_heads=model.config.num_key_value_heads,
                          num_section_types=8, rope_pair_dim=16,
                          max_offset=0.03, init_log_scale=-3.5, strength=1.0)
        .to(dtype).to(layer12.self_attn_norm.weight.device)
    )
    layer12.section_rope_time_dim = 32
    layer12.use_phase_memory = False


def load_section_rope_adapter(model) -> int:
    """Load Section-RoPE weights from checkpoint, return keys loaded."""
    # Try adapter first (new format), fall back to full decoder checkpoint
    adapter_path = SECTION_ROPE_DIR / "adapter_model.safetensors"
    if adapter_path.is_file():
        sd = safetensors.torch.load_file(str(adapter_path))
    else:
        full_path = SECTION_ROPE_DIR / "model.safetensors"
        if not full_path.is_file():
            return 0
        sd = safetensors.torch.load_file(str(full_path))
        sd = {k: v for k, v in sd.items() if "section_rope_offset" in k}

    # Keys in checkpoint: "layers.12.section_rope_offset_module.xxx"
    # Keys in model:      "decoder.layers.12.section_rope_offset_module.xxx"
    model_sd = {"decoder." + k: v for k, v in sd.items()}
    model.load_state_dict(model_sd, strict=False)
    return len(model_sd)


def run_condition(dit_handler, llm_handler, params, config,
                  lyrics_text: str, condition: str) -> dict | None:
    """Run generation, capture attention, return metrics.

    condition: "baseline" | "real" | "unknown" | "shuffled"
    """
    layer12 = dit_handler.model.decoder.layers[12]
    offset_module = getattr(layer12, "section_rope_offset_module", None)

    # ---- Set up condition-specific overrides ----
    if condition == "baseline":
        layer12.use_section_rope = False

    if condition == "unknown":
        dit_handler.model._lyrics_raw = re.sub(r"\[.*?\]", "", lyrics_text).strip()
    else:
        dit_handler.model._lyrics_raw = lyrics_text

    capture = AttentionCapture(dit_handler.model)

    shuffled_orig_forward = None
    if condition == "shuffled" and offset_module is not None:
        shuffled_orig_forward = offset_module.forward
        def _make_shuffled(orig_fn):
            def _shuffled(section_ids):
                B, L = section_ids.shape
                perm = torch.stack([
                    torch.randperm(L, device=section_ids.device)
                    for _ in range(B)
                ], dim=0)
                return orig_fn(section_ids.gather(1, perm))
            return _shuffled
        offset_module.forward = _make_shuffled(shuffled_orig_forward)

    # ---- Run ----
    with capture:
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=str(OUTPUT_DIR),
        )

    # ---- Restore ----
    if condition == "baseline":
        layer12.use_section_rope = True
    if condition == "unknown":
        dit_handler.model._lyrics_raw = lyrics_text
    if condition == "shuffled" and shuffled_orig_forward is not None:
        offset_module.forward = shuffled_orig_forward

    if not result.success:
        print(f"  ❌ {condition}: {result.error}")
        return None

    attn = capture.tensor
    if attn is None:
        print(f"  ⚠ {condition}: no attention captured")
        return None

    # ---- Parse section_ids from original lyrics (real labels) ----
    L = attn.shape[-1]
    parsed = LyricsStructureParser().parse(lyrics_text, num_chunks=L)
    section_ids = parsed.section_type_ids.unsqueeze(0)  # [1, L]

    metrics = compute_attention_metrics(attn, section_ids)
    metrics["condition"] = condition
    metrics["L"] = L
    metrics["T"] = attn.shape[-2]

    # Save raw attention for later inspection
    save_path = OUTPUT_DIR / f"attn_{condition}.pt"
    torch.save({
        "attention": attn,
        "section_ids": section_ids,
        "section_names": SECTION_NAMES,
    }, save_path)
    print(f"  ✓ {condition}: attention saved -> {save_path.name}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Section-RoPE Cross-Attention Diagnostic")
    print("=" * 70)

    # ---- Load base model ----
    print("\n[1/4] Loading base model...")
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    print("  ✓ Base model loaded")

    # ---- Inject Section-RoPE + adapter ----
    print("\n[2/4] Injecting Section-RoPE...")
    inject_section_rope(
        dit_handler.model,
        dit_handler.model.dtype,
        dit_handler.model.device,
    )
    n_keys = load_section_rope_adapter(dit_handler.model)
    print(f"  ✓ Section-RoPE adapter loaded ({n_keys} keys)")

    dit_handler.model.eval()

    # ---- Init LM ----
    print("\n[3/4] Initializing 5Hz LM...")
    ok = llm_handler.initialize(
        checkpoint_dir=str(MODEL_ROOT),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )
    if not ok:
        print("  ❌ LM init failed")
        sys.exit(1)
    print("  ✓ LM ready")

    # ---- Config ----
    print("\n[4/4] Configuring generation...")
    params = GenerationParams(
        task_type="text2music",
        caption="ballad, pop, female vocal, piano, strings, drums, "
                "romantic, melancholic, 137 bpm, E major",
        lyrics=LYRICS,
        instrumental=False,
        bpm=137,
        keyscale="E major",
        timesignature="4",
        vocal_language="zh",
        duration=60,
        inference_steps=25,
        guidance_scale=7.0,
        seed=42,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format="flac",
        use_random_seed=False,
    )

    # ---- Run conditions ----
    CONDITIONS = ["baseline", "real", "unknown", "shuffled"]
    all_metrics = {}

    for cond in CONDITIONS:
        print(f"\n{'─' * 70}")
        print(f"Condition: {cond.upper()}")
        print(f"{'─' * 70}")
        metrics = run_condition(
            dit_handler, llm_handler, params, config, LYRICS, cond,
        )
        all_metrics[cond] = metrics

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    header = (f"{'Condition':<12} {'Entropy':<10} {'NormEnt':<10} "
              f"{'SecVar':<12} {'Stability':<10}")
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        m = all_metrics.get(cond)
        if m is None:
            print(f"{cond:<12} {'FAILED':<10} {'':<10} {'':<12} {'':<10}")
            continue
        print(f"{cond:<12} {m['entropy']:<10} {m['norm_entropy']:<10} "
              f"{m['section_var']:<12} {m['stability']:<10}")
        sec_frac = m.get("section_frac", {})
        parts = "  ".join(f"{k}={v:.3f}" for k, v in sec_frac.items())
        print(f"  section frac: {parts}")

    # ---- Save results ----
    results_path = OUTPUT_DIR / "attention_diagnostic_results.json"
    serializable = {}
    for cond, m in all_metrics.items():
        if m is not None:
            serializable[cond] = {k: v for k, v in m.items()
                                  if isinstance(v, (str, int, float, dict))}
    results_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
