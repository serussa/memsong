#!/usr/bin/env python3
"""
ACE-Step 1.5 文本到音乐生成 —— PMDC Residual Clock 版

支持已训练的 PMDC clock 的 attention intervention，以及 static clean_parser_no_control。

用法
----
    # Static clean_parser_no_control（免训练）
    python acestep/run_inference_pmdc.py

    # 已训练的 PMDC clock
    python acestep/run_inference_pmdc.py --pmdc-ckpt /path/to/pmdc_clock.pt

    # Ablation
    python acestep/run_inference_pmdc.py --pmdc-ckpt ... --force-p-final-base
    python acestep/run_inference_pmdc.py --pmdc-ckpt ... --duration-bias-off
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.tgca.lyrics_parser import LyricsStructureParser
from acestep.phase_memory import (
    PMDCResidualClock,
    parse_lyrics_to_units,
    build_duration_scaffold,
    build_duration_interval_bias,
)


def load_pmdc_clock(ckpt_path: str, hidden_size: int, device: torch.device, dtype: torch.dtype):
    """Load trained PMDCResidualClock from checkpoint.

    The clock always runs in float32 for numerical stability.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("pmdc_config", {})
    clock = PMDCResidualClock(
        dim=hidden_size,
        hidden_dim=cfg.get("hidden_dim", 128),
        beta_init=cfg.get("beta_init", 0.05),
        beta_max=cfg.get("beta_max", 0.15),
        use_delta_h=cfg.get("use_delta_h", True),
    )
    # Load state dict (allow bf16 → f32 conversion)
    sd = ckpt["clock_state_dict"]
    sd = {k: v.float() for k, v in sd.items()}  # convert to float32
    clock.load_state_dict(sd)
    clock = clock.to(device).to(torch.float32)
    clock.eval()
    gate_logit = ckpt.get("pmdc_gate_logit", torch.tensor(0.0))
    gate = torch.sigmoid(gate_logit).item()
    print(f"  [PMDC] Loaded from {ckpt_path}")
    print(f"  [PMDC] gate={gate:.4f}, beta={clock.beta.item():.4f}")
    return clock, gate


def build_scaffold(lyrics_text: str, L_text: int, device: torch.device):
    """Parse lyrics with tag-aware parser → build scaffold."""
    parser = LyricsStructureParser()
    section_ids = parser.parse(lyrics_text, num_chunks=L_text).section_type_ids
    units, _, debug = parse_lyrics_to_units(lyrics_text, section_ids)
    tcm = debug.get("tag_control_mask", None)
    scaffold = build_duration_scaffold(units, text_len=L_text, tag_control_mask=tcm)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in scaffold.items()}


def warmup_and_patch(
    model,
    pmdc_clock,
    scaffold,
    gate_val,
    sigma,
    lambda_,
    max_bias,
    force_p_final_base=False,
    duration_bias_off=False,
):
    """
    Warmup: run one decoder forward → collect layer 12 H → clock → p_final → bias.
    Then patch ``eager_attention_forward`` for the rest of generation.

    Returns a ``restore`` callable that undoes the patch.
    """
    if duration_bias_off:
        return lambda: None  # no-op

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    D = model.config.hidden_size

    # ---- 1. Warmup forward to get layer 12 H -------------------------------
    # 构造一个假的 latent 输入：纯噪声
    # DiT patch_size=2 所以 hidden T = latent T / 2
    T_target = int(args.duration * 50) if hasattr(args, 'duration') else 3000

    B = 1
    fake_xt = torch.randn(B, T_target, 64, device=device, dtype=dtype)
    fake_am = torch.ones(B, T_target, device=device, dtype=dtype)
    fake_t = torch.full((B,), 0.5, device=device, dtype=dtype)

    # 用 silence_latent 作为 context_latents
    silence_latent = getattr(model, "silence_latent", None)
    if silence_latent is None:
        silence_latent = torch.zeros(B, T_target, 128, device=device, dtype=dtype)
    else:
        silence_latent = silence_latent.expand(B, T_target, -1)

    # encoder_hidden_states: 用 model 自己的 null_condition_emb
    null_emb = getattr(model, "null_condition_emb", None)
    if null_emb is not None:
        # null_condition_emb may be [B, 1, D] — expand along second dim
        dummy_eh = null_emb.expand(B, 769, -1)
        dummy_ea = torch.ones(B, 769, device=device, dtype=dtype)
    else:
        dummy_eh = torch.randn(B, 769, 2048, device=device, dtype=dtype)
        dummy_ea = torch.ones(B, 769, device=device, dtype=dtype)

    # hook
    hs_list = []

    def _hook(m, i, o):
        hs_list.append(o[0])

    handle = model.decoder.layers[12].register_forward_hook(_hook)
    with torch.no_grad():
        model.decoder(
            hidden_states=fake_xt,
            timestep=fake_t,
            timestep_r=fake_t,
            attention_mask=fake_am,
            encoder_hidden_states=dummy_eh,
            encoder_attention_mask=dummy_ea,
            context_latents=silence_latent,
            use_cache=False,
            output_attentions=False,
        )
    handle.remove()

    if hs_list:
        H_warm = hs_list[0].float()  # [1, T_hidden, D]
        T_pmdc = H_warm.shape[1]
    else:
        print("  [WARN] Warmup hook returned nothing, using fallback")
        T_pmdc = T_target
        H_warm = torch.zeros(1, T_pmdc, D, device="cpu")

    # ---- 2. PMDC clock forward (or linear p_base) -------------------------
    p_base = torch.linspace(0, 1, T_pmdc, device=device).float().unsqueeze(0)

    if pmdc_clock is not None and not force_p_final_base:
        with torch.no_grad():
            H_clock = H_warm.float()
            print(f"  H_clock: dtype={H_clock.dtype}, device={H_clock.device}, shape={H_clock.shape}")
            print(f"  p_base: dtype={p_base.dtype}, device={p_base.device}, shape={p_base.shape}")
            _, p_final, _ = pmdc_clock(H_clock, p_base)
        print(f"  [PMDC] p_final range: [{p_final.min().item():.4f}, {p_final.max().item():.4f}], "
              f"p_delta={(p_final-p_base).abs().mean().item():.6f}")
    else:
        p_final = p_base

    # ---- 3. Build duration bias -------------------------------------------
    bias = build_duration_interval_bias(
        p_final=p_final.to(device),
        unit_boundaries=scaffold["unit_boundaries"],
        token_to_unit=scaffold["token_to_unit"],
        attendable_mask=scaffold.get("attendable_mask", scaffold["lyric_mask"]),
        sigma=sigma, lambda_=lambda_, max_bias=max_bias,
    ).unsqueeze(1)  # [1, 1, T, L]

    # Ensure bias dtype matches model
    if dtype == torch.bfloat16:
        bias = bias.to(torch.bfloat16)

    # ---- 4. Patch eager_attention_forward ---------------------------------
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    import sys

    ca_module = model.decoder.layers[12].cross_attn
    cls = type(ca_module)
    attn_mod = sys.modules[cls.__module__]
    orig_eaf = getattr(attn_mod, "eager_attention_forward", None)

    amask = scaffold.get("attendable_mask", scaffold["lyric_mask"])
    if amask.dim() == 1:
        amask = amask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def _patched_forward(*fargs, **fkwargs):
        module = fargs[0]
        q = fargs[1]; k = fargs[2]; v = fargs[3]
        am = fargs[4] if len(fargs) > 4 else fkwargs.get("attention_mask")
        sc = fkwargs.get("scaling", fargs[5] if len(fargs) > 5 else None)
        dr = fkwargs.get("dropout", 0.0)

        ks = repeat_kv(k, module.num_key_value_groups)
        vs = repeat_kv(v, module.num_key_value_groups)
        aw = torch.matmul(q, ks.transpose(2, 3)) * sc
        if am is not None and isinstance(am, torch.Tensor):
            aw = aw + am[:, :, :, :ks.shape[-2]]

        # Broadcast mask to batch size of current forward
        b_cur = aw.shape[0]
        amask_cur = amask
        if amask_cur.shape[0] != b_cur:
            bf = b_cur // max(amask_cur.shape[0], 1)
            amask_cur = amask_cur.repeat(bf, 1, 1, 1) if bf > 1 else amask_cur
        amask_cur = amask_cur.bool()

        # Broadcast bias to current batch size
        bias_cur = bias
        if bias_cur.shape[0] != b_cur:
            bf = b_cur // max(bias_cur.shape[0], 1)
            bias_cur = bias_cur.repeat(bf, 1, 1, 1) if bf > 1 else bias_cur

        text_float = amask_cur.float()
        attn_base = F.softmax(aw, dim=-1, dtype=torch.float32)
        text_mass_base = (attn_base * text_float).sum(dim=-1)

        text_logits = (aw + gate_val * bias_cur).masked_fill(~amask_cur, float("-inf"))
        attn_text = F.softmax(text_logits, dim=-1, dtype=torch.float32)
        attn_text = attn_text.masked_fill(~amask_cur, 0.0)

        non_text_logits = aw.masked_fill(amask_cur, float("-inf"))
        attn_non_text = F.softmax(non_text_logits, dim=-1, dtype=torch.float32)
        attn_non_text = attn_non_text.masked_fill(amask_cur, 0.0)

        aw = (attn_text * text_mass_base.unsqueeze(-1) +
              attn_non_text * (1.0 - text_mass_base).unsqueeze(-1))
        aw = aw / (aw.sum(dim=-1, keepdim=True) + 1e-10)
        aw = aw.to(q.dtype)

        aw = F.dropout(aw, p=dr, training=module.training)
        return torch.matmul(aw, vs).transpose(1, 2).contiguous(), aw

    setattr(attn_mod, "eager_attention_forward", _patched_forward)
    print(f"  [Patch] attention patched (gate={gate_val:.4f})")

    # Return restore function
    def restore():
        setattr(attn_mod, "eager_attention_forward", orig_eaf)

    return restore


def main():
    global args
    parser = argparse.ArgumentParser(description="ACE-Step PMDC Inference")
    parser.add_argument("--pmdc-ckpt", type=str, default=None)
    parser.add_argument("--force-p-final-base", action="store_true")
    parser.add_argument("--duration-bias-off", action="store_true")
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--lambda", type=float, dest="lambda_", default=0.5)
    parser.add_argument("--gate", type=float, default=0.35)
    parser.add_argument("--max-bias", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=240)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--caption", type=str, default="")
    parser.add_argument("--lyrics", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=str(ACE_STEP_ROOT / "output"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print("ACE-Step 1.5 PMDC 生成")
    print("=" * 70)
    if args.pmdc_ckpt:
        print(f"  模式: trained PMDC ({args.pmdc_ckpt})")
    else:
        print("  模式: static clean_parser_no_control")
    print(f"  sigma={args.sigma}, lambda={args.lambda_}, gate={args.gate}")

    # ---- 默认歌词和 prompt ----
    if not args.caption:
        args.caption = "ballad, pop, female vocal, piano, strings, drums, romantic, melancholic, 137 bpm, E major"
    if not args.lyrics:
        args.lyrics = r"""[Intro]

[Verse]
爱总忽然退潮
心慌乱触礁
沉没在深海里
看海面闪耀
但回忆像水草
紧紧的缠绕
梦才温热眼角
就冰冷掉
努力越过风暴
向着未来飘
我们才会遇到
感动的拥抱
你总是能知道
我的坚强剩多少

[Pre-chorus]
给我最刚好的依靠

[Chorus]
你手心的太阳
只轻放在我背上
委屈就能笑着落泪
被释放
在手心的太阳
黑暗里特别明亮
让远路好像
是一种分享
而不是漫长

[Inst]
让眼睛看不到
嫉妒的燃烧

[Verse]
让耳朵听不到
谎言的吵闹
再没有人相信
爱能永恒那一秒
我们正坚定的微笑

[Chorus]
你手心的太阳
有种安定的力量
就算世界再乱我也
不心慌
我手心的太阳
或许只像个月亮
却用所有爱
为你投射我最暖的光芒
你手心的太阳
只轻放在我背上

[Bridge]
委屈就能笑着落泪
被释放
在手心的太阳
黑暗里特别明亮
让远路好像
是一种分享
你手心的太阳
有种安定的力量

[Chorus]
就算世界再乱我也
不心慌
我手心的太阳
或许只像个月亮
却用所有爱
为你投射我
最暖的光芒"""

    # ========== 1-3. 初始化模型 ==========
    print("\n[1/5] 初始化处理器...")
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    print("\n[2/5] 初始化 DiT...")
    dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device=args.device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    model = dit_handler.model.eval()
    model.config.use_section_rope_offset = False
    for lm in model.decoder.layers:
        if getattr(lm, "use_section_rope", False):
            lm.use_section_rope = False
        if getattr(lm, "use_phase_memory", False):
            lm.use_phase_memory = False

    # CRITICAL: force eager attention so our patch runs
    model.config._attn_implementation = "eager"
    model.config._attn_implementation_compiled = None

    print("\n[3/5] 初始化 5Hz LM...")
    llm_handler.initialize(
        checkpoint_dir=str(MODEL_ROOT),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device=args.device,
    )

    # ========== 4. 加载 PMDC clock ==========
    clock = None
    gate_val = args.gate
    if args.pmdc_ckpt:
        print("\n[4/5] 加载 PMDC Clock...")
        device = torch.device(args.device)
        dtype = torch.bfloat16
        clock, trained_gate = load_pmdc_clock(args.pmdc_ckpt, model.config.hidden_size, device, dtype)
        if not args.force_p_final_base:
            gate_val = trained_gate

    # ========== 5. 预计算 scaffold ==========
    # 先走一遍 encoder 拿到 L_text
    print("\n[4.5/5] 预计算 scaffold...")
    lyrics = args.lyrics
    parser = LyricsStructureParser()
    # 估计 L_text = 769（大多数歌词差不多），建一个用于 bias 的 scaffold
    L_est = 769
    scaffold = build_scaffold(lyrics, L_est, torch.device(args.device))
    print(f"  Scaffold: {scaffold['unit_boundaries'].shape[-1]-1} units, "
          f"lyric_mask={scaffold['lyric_mask'].sum().item()}/{L_est}")

    # ========== 6. Warmup + patch ==========
    restore_fn = lambda: None
    if not args.duration_bias_off:
        print("\n[5/5] Warmup + patch attention...")
        restore_fn = warmup_and_patch(
            model, clock, scaffold, gate_val,
            args.sigma, args.lambda_, args.max_bias,
            force_p_final_base=args.force_p_final_base,
            duration_bias_off=args.duration_bias_off,
        )
    else:
        print("\n[5/5] Duration bias OFF — 原始模型生成")

    # ========== 7. 生成 ==========
    print(f"\n  生成 {args.duration}s × {args.steps} steps...")
    params = GenerationParams(
        task_type="text2music",
        caption=args.caption,
        lyrics=lyrics,
        instrumental=False,
        bpm=137,
        keyscale="E major",
        timesignature="4",
        vocal_language="zh",
        duration=args.duration,
        inference_steps=args.steps,
        guidance_scale=7.0,
        seed=args.seed,
        thinking=True,
        use_cot_metas=True,
        use_cot_caption=True,
        lm_temperature=0.75,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format="flac",
        use_random_seed=False,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=str(output_dir),
    )

    # 恢复
    restore_fn()

    if result.success:
        print(f"\n✓ 生成成功!")
        for a in result.audios:
            print(f"   {a['path']}")
    else:
        print(f"\n✗ 失败: {result.error}")


if __name__ == "__main__":
    main()
