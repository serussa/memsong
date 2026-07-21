#!/usr/bin/env python3
"""
ACE-Step PM-CTR 实验批处理脚本

对 prompts_dir 中每个 lyrics 文件，对每个 seed，对每个 method 生成音频并保存 diagnostics。

目录结构:
  eval_outputs/
    prompt_001/
      seed_1234/
        baseline/
        softmax_retrieval/
        lyric_only_transport/
        full_pmctr/
        qk_scale_0/

每个 method 目录保存:
  generated.flac
  inference_config.json
  transport_units.csv       (transport 方法)
  transport_summary.json    (transport 方法)
  transport_heatmap.png     (transport 方法)
  condition_usage_histogram.png (transport 方法)
  lyric_center_curve.png    (transport 方法)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(ACE_STEP_ROOT))

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.run_pmctr_eval import load_adapter, install_hooks, run_inference


DEFAULT_METHODS = [
    "baseline",
    "softmax_retrieval",
    "lyric_only_transport",
    "full_pmctr",
    "qk_scale_0",
]

DEFAULT_SEEDS = [1234, 2345, 3456]

# Default shared config
DEFAULT_CAPTION = "ballad, pop, female vocal, piano, strings, drums, romantic, melancholic, 120 bpm, C major"
DEFAULT_BPM = 120
DEFAULT_KEY = "C major"
DEFAULT_DURATION = 120
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 7.0
DEFAULT_VOCAL_LANG = "zh"


def load_lyrics_from_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="ACE-Step PM-CTR Batch Eval Suite")
    parser.add_argument("--prompts-dir", type=str, default=None,
                        help="Directory containing lyrics .txt files. "
                             "If not given, uses a single default prompt.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                        choices=["baseline", "lyric_only_transport", "full_pmctr",
                                 "softmax_retrieval", "qk_scale_0"])
    parser.add_argument("--checkpoint-path", type=str,
                        default="/tmp/transport_1epoch_v5_control/final/pm_retrieval.pt")
    parser.add_argument("--output-root", type=str, default="./eval_outputs")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="Duration in seconds")
    parser.add_argument("--caption", type=str, default=DEFAULT_CAPTION)
    parser.add_argument("--bpm", type=int, default=DEFAULT_BPM)
    parser.add_argument("--key", type=str, default=DEFAULT_KEY)
    parser.add_argument("--vocal-lang", type=str, default=DEFAULT_VOCAL_LANG)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--no-llm", action="store_true", default=True)

    args = parser.parse_args()

    # ========== Collect prompts ==========
    if args.prompts_dir and os.path.isdir(args.prompts_dir):
        prompt_files = sorted(Path(args.prompts_dir).glob("*.txt"))
        if not prompt_files:
            print(f"[ERROR] No .txt files found in {args.prompts_dir}")
            sys.exit(1)
        prompts = []
        for pf in prompt_files:
            lyrics = load_lyrics_from_file(str(pf))
            prompts.append((pf.stem, lyrics))
        print(f"[INFO] Loaded {len(prompts)} prompts from {args.prompts_dir}")
    else:
        # Single default prompt
        default_lyrics = """[INTRO]

[VERSE]
爱总忽然退潮 心慌乱触礁
沉没在深海里 看海面闪耀
但回忆像水草 紧紧的缠绕
梦才温热眼角 就冰冷掉
努力越过风暴 向着未来飘
我们才会遇到 感动的拥抱
你总是能知道 我的坚强剩多少

[CHORUS]
你手心的太阳 只轻放在我背上
委屈就能笑着落泪 被释放
在手心的太阳 黑暗里特别明亮
让远路好像 是一种分享 而不是漫长

[OUTRO]
"""
        prompts = [("prompt_001", default_lyrics)]
        print(f"[INFO] Using single default prompt")

    seeds = args.seeds
    methods = args.methods
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)

    total_jobs = len(prompts) * len(seeds) * len(methods)
    print(f"\n[INFO] Batch eval plan: {len(prompts)} prompts × {len(seeds)} seeds × {len(methods)} methods = {total_jobs} jobs")

    # ========== Init services once ==========
    print("\n[1/3] Initializing handlers...")
    dit_handler = AceStepHandler()
    llm_handler = None

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

    print(f"\n[3/3] Running {total_jobs} evaluations...")
    print("=" * 70)

    completed = 0
    failed = 0

    for prompt_id, (prompt_name, lyrics_text) in enumerate(prompts):
        for seed in seeds:
            for method in methods:
                prompt_dir = os.path.join(output_root, f"{prompt_name}")
                seed_dir = os.path.join(prompt_dir, f"seed_{seed}")
                method_dir = os.path.join(seed_dir, method)
                os.makedirs(method_dir, exist_ok=True)

                print(f"\n[{completed + 1}/{total_jobs}] {prompt_name} seed={seed} method={method}")

                # Build params
                params = GenerationParams(
                    task_type="text2music",
                    caption=args.caption,
                    lyrics=lyrics_text,
                    instrumental=False,
                    bpm=args.bpm,
                    keyscale=args.key,
                    timesignature="4",
                    vocal_language=args.vocal_lang,
                    duration=args.duration,
                    inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    seed=seed,
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

                t0 = time.time()
                result = run_inference(
                    dit_handler=dit_handler,
                    llm_handler=llm_handler,
                    params=params,
                    config=gen_config,
                    method=method,
                    ckpt_path=args.checkpoint_path,
                    output_dir=method_dir,
                    sample_id=prompt_id,
                    seed=seed,
                    save_transport_diagnostics=True,
                    save_heatmap=True,
                )
                elapsed = time.time() - t0

                if result.get("success", False):
                    completed += 1
                    print(f"  [OK] {elapsed:.1f}s -> {result.get('audio_path', '?')}")
                else:
                    failed += 1
                    print(f"  [FAIL] {result.get('error', 'unknown')}")

                # Write a quick marker
                marker = {
                    "prompt": prompt_name,
                    "seed": seed,
                    "method": method,
                    "success": result.get("success", False),
                    "elapsed_sec": round(elapsed, 1),
                    "audio_path": result.get("audio_path"),
                }
                with open(os.path.join(method_dir, ".marker.json"), "w") as f:
                    json.dump(marker, f, indent=2)

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print(f"Batch eval complete: {completed} OK, {failed} FAIL, {total_jobs} total")
    print(f"Output root: {output_root}")
    print("=" * 70)

    # Write master summary
    summary = {
        "total_jobs": total_jobs,
        "completed": completed,
        "failed": failed,
        "prompts": len(prompts),
        "seeds": seeds,
        "methods": methods,
        "checkpoint": args.checkpoint_path,
        "duration": args.duration,
        "steps": args.steps,
    }
    with open(os.path.join(output_root, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
