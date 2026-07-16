#!/usr/bin/env python3
"""Quick runner: band ablation at 270s for 3 critical bands."""
import sys, os, json
from pathlib import Path

SRC = Path("/root/ACE-Step-1.5")
OUT = SRC / "output" / "rope_exp3" / "dur_270s"
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
sys.path.insert(0, str(SRC))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
sys.path.insert(0, str(SRC / "scripts"))
from exp_rope_band_ablation import run_band_condition

# Only the 3 most informative bands
LONG_BANDS = [
    ("Full",   0, 64),
    ("MidLow", 32, 48),
    ("NoRoPE",  0,  0),
]

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Band Ablation @ 270s (3 critical bands)")
print("=" * 60)

print("\n[1/3] Loading model...")
dit = AceStepHandler()
dit.initialize_service(
    project_root=str(MODEL_ROOT),
    config_path="acestep-v15-sft",
    device="cuda",
    use_flash_attention=False,
    compile_model=False,
    offload_to_cpu=False,
)
model = dit.model.eval()
model.config.use_section_rope_offset = False
model.config._attn_implementation = "eager"
for l in model.decoder.layers:
    if getattr(l, "use_section_rope", False): l.use_section_rope = False
    if getattr(l, "use_phase_memory", False): l.use_phase_memory = False

print("[2/3] Initializing 5Hz LM...")
llm = LLMHandler()
llm.initialize(checkpoint_dir=str(MODEL_ROOT), lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")

print("[3/3] Running 270s bands...")
all_results = {}
for band_name, start, end in LONG_BANDS:
    if hasattr(dit.model.decoder.rotary_emb, '_original_forward'):
        dit.model.decoder.rotary_emb.forward = dit.model.decoder.rotary_emb._original_forward

    summary = run_band_condition(dit, llm, band_name, start, end,
                                 270.0, 25, 42, OUT)
    all_results[band_name] = summary
    print(f"  {band_name}: full_delta={summary.get('avg_full_delta', 'N/A')}")

print("\n" + "=" * 60)
print("270s BAND ABLATION SUMMARY")
print("=" * 60)
print(f"{'Band':<10} {'FullDelta':>10} {'SlidDelta':>10} {'HeadEnt':>8} {'TailEnt':>8}")
print("-" * 46)
for band_name, _, _ in LONG_BANDS:
    s = all_results.get(band_name, {})
    if not s or s.get("n_captured", 0) == 0:
        print(f"{band_name:<10} {'FAIL':>10}")
        continue
    print(f"{band_name:<10} {s.get('avg_full_delta', 0):>+10.4f} "
          f"{s.get('avg_sliding_delta', 0):>+10.4f} "
          f"{s.get('avg_head_entropy', 0):>8.4f} "
          f"{s.get('avg_tail_entropy', 0):>8.4f}")

combined = {}
for k, v in all_results.items():
    if v:
        combined[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, type(None)))}
(out_json := OUT / "band_ablation_270s.json").write_text(json.dumps(combined, indent=2))
print(f"\nSaved to {out_json}")
print("Done.")
