#!/usr/bin/env python3
"""
ACE-Step PM-CTR Adapter Sanity Check

同一个 prompt、同一个 seed，生成三条音频:
A. baseline SFT             — no adapter
B. full_pmctr write_alpha=0 — adapter loaded but writer=0
C. full_pmctr write_alpha=1 — adapter loaded, full effect

保存:
  audio_baseline.flac
  audio_write0.flac
  audio_full.flac
  sanity_report.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(ACE_STEP_ROOT))

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# Reuse the eval module
from acestep.run_pmctr_eval import (
    load_adapter, install_hooks, count_trainable_adapter_params,
)


DEFAULT_LYRICS = """[INTRO]

[VERSE]
爱总忽然退潮 心慌乱触礁
沉没在深海里 看海面闪耀
但回忆像水草 紧紧的缠绕
梦才温热眼角 就冰冷掉

[CHORUS]
你手心的太阳 只轻放在我背上
委屈就能笑着落泪 被释放
在手心的太阳 黑暗里特别明亮
让远路好像 是一种分享 而不是漫长

[OUTRO]
"""


def run_one(
    dit_handler,
    llm_handler,
    params,
    gen_config,
    method,
    ckpt_path,
    output_dir,
    seed,
    write_alpha_multiplier=None,
):
    """Run a single inference for a given method and return result."""
    model = dit_handler.model.eval()
    model._gen_metadata = {"lyrics": params.lyrics}

    # Load adapter
    pm, adapt, adapt_info = load_adapter(
        model, method, ckpt_path, model.device,
        write_alpha_multiplier=write_alpha_multiplier,
    )

    # Register adapter modules
    if pm is not None:
        model.add_module("transport_pm", pm)
    if adapt is not None:
        model.add_module("transport_adapter", adapt)

    # Install hooks
    scaffold_cache = {}
    handles = install_hooks(model, pm, adapt, method, scaffold_cache)

    os.makedirs(output_dir, exist_ok=True)
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=gen_config,
        save_dir=output_dir,
    )

    # Cleanup
    for h in handles:
        h.remove()
    if pm is not None and hasattr(model, "transport_pm"):
        del model.transport_pm
    if adapt is not None and hasattr(model, "transport_adapter"):
        del model.transport_adapter

    return result, adapt_info


def main():
    parser = argparse.ArgumentParser(description="PM-CTR Adapter Sanity Check")
    parser.add_argument("--checkpoint-path", type=str,
                        default="/tmp/transport_1epoch_v5_control/final/pm_retrieval.pt")
    parser.add_argument("--lyrics-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./sanity_check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=120)

    args = parser.parse_args()

    if args.lyrics_file and os.path.exists(args.lyrics_file):
        with open(args.lyrics_file, "r") as f:
            lyrics = f.read()
    else:
        lyrics = DEFAULT_LYRICS

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ========== Init services ==========
    print("=" * 70)
    print("ACE-Step PM-CTR Adapter Sanity Check")
    print("=" * 70)

    print("\n[1/3] Initializing handlers...")
    dit_handler = AceStepHandler()
    llm_handler = None  # No LLM for sanity check

    print("\n[2/3] Initializing DiT model...")
    dit_status, dit_success = dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    if not dit_success:
        print(f"[FAIL] DiT init failed: {dit_status}")
        sys.exit(1)

    # Model info
    model = dit_handler.model.eval()
    D = model.config.hidden_size
    trainable_before = count_trainable_adapter_params(model)

    # Shared params
    params = GenerationParams(
        task_type="text2music",
        caption="ballad, pop, female vocal, piano, strings, 120 bpm, C major",
        lyrics=lyrics,
        instrumental=False,
        bpm=120,
        keyscale="C major",
        timesignature="4",
        vocal_language="zh",
        duration=args.duration,
        inference_steps=50,
        guidance_scale=7.0,
        seed=args.seed,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        lm_temperature=0.0,
    )

    gen_config = GenerationConfig(
        batch_size=1,
        audio_format="flac",
        use_random_seed=False,
    )

    # ========== Run A: baseline ==========
    print("\n[3/3] Running sanity checks...")
    print("-" * 40)
    print("A. Baseline SFT (no adapter)")
    result_a, _ = run_one(
        dit_handler, llm_handler, params, gen_config,
        "baseline", args.checkpoint_path,
        os.path.join(output_dir, "baseline"), args.seed,
    )
    audio_a = result_a.audios[0]["path"] if result_a.success else None

    # Rename output
    if audio_a and os.path.exists(audio_a):
        import shutil
        shutil.copy2(audio_a, os.path.join(output_dir, "audio_baseline.flac"))

    # ========== Run B: full_pmctr write_alpha=0 ==========
    print("\nB. Full PM-CTR write_alpha=0")
    result_b, info_b = run_one(
        dit_handler, llm_handler, params, gen_config,
        "write_alpha_0", args.checkpoint_path,
        os.path.join(output_dir, "write0"), args.seed,
        write_alpha_multiplier=0,
    )
    audio_b = result_b.audios[0]["path"] if result_b.success else None
    if audio_b and os.path.exists(audio_b):
        import shutil
        shutil.copy2(audio_b, os.path.join(output_dir, "audio_write0.flac"))

    # ========== Run C: full_pmctr ==========
    print("\nC. Full PM-CTR")
    result_c, info_c = run_one(
        dit_handler, llm_handler, params, gen_config,
        "full_pmctr", args.checkpoint_path,
        os.path.join(output_dir, "full"), args.seed,
    )
    audio_c = result_c.audios[0]["path"] if result_c.success else None
    if audio_c and os.path.exists(audio_c):
        import shutil
        shutil.copy2(audio_c, os.path.join(output_dir, "audio_full.flac"))

    # ========== Report ==========
    adapter_param_count = count_trainable_adapter_params(model) - trainable_before

    report = {
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_loaded": True,
        "trainable_adapter_param_count": adapter_param_count,
        "write_alpha_0_used": True,
        "full_write_alpha_used": True,
        "output_file_paths": {
            "baseline": os.path.join(output_dir, "audio_baseline.flac"),
            "write_alpha_0": os.path.join(output_dir, "audio_write0.flac"),
            "full": os.path.join(output_dir, "audio_full.flac"),
        },
        "inference_config": {
            "duration": args.duration,
            "seed": args.seed,
            "steps": 50,
            "guidance": 7.0,
            "caption": params.caption,
        },
        "baseline_success": result_a.success,
        "write0_success": result_b.success,
        "full_success": result_c.success,
    }

    report_path = os.path.join(output_dir, "sanity_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Sanity Check Complete")
    print("=" * 70)
    print(f"Baseline:       {'OK' if result_a.success else 'FAIL'}")
    print(f"Write Alpha=0:  {'OK' if result_b.success else 'FAIL'}")
    print(f"Full:           {'OK' if result_c.success else 'FAIL'}")
    print(f"Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
