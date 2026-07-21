#!/usr/bin/env python3
"""
ACE-Step 1.5 文本到音乐生成脚本 + PM Sinkhorn Transport Retrieval
基于 run_inference.py，仅添加 adapter 加载 + hook 注入。
prompt、参数、flow 结构完全相同。
"""

import argparse
import os
import sys
from pathlib import Path

# 最新训练的 position_only + PM gate checkpoint（loss=0.53, 2000步）
BEST_CHECKPOINT = "/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt"

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music


def _inject_retrieval(model, lyrics_text, T_eff, checkpoint_path, use_pm_gate, device):
    """从 checkpoint 加载 PM + adapter，通过 prepare_condition patch 注册 layer-12 hook。"""
    from acestep.phase_memory import (
        TransportRetrievalAdapter, PMRetrievalPhaseMemory,
        parse_lyrics_to_units, build_duration_scaffold,
    )
    from acestep.tgca.lyrics_parser import LyricsStructureParser

    D = model.config.hidden_size
    pm = PMRetrievalPhaseMemory(
        dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True,
    ).to(device).float()
    adapt = TransportRetrievalAdapter(
        hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        transport_mode="sinkhorn", sinkhorn_iters=10,
        transport_sigma=0.18,
        scoring_mode="position_only",
        use_pm_gate=use_pm_gate, gate_hidden_dim=128,
        write_alpha_init=0.005, write_alpha_max=0.01,
        out_proj_init_std=0.01,
    ).to(device).float()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Debug: print checkpoint write_alpha
    if "retrieval_adapter" in ckpt:
        ra = ckpt["retrieval_adapter"]
        wl = ra.get("write_logit", None)
        if wl is not None:
            from math import exp as _exp
            sig = 1.0 / (1.0 + _exp(-wl.item()))
            print(f"  [DEBUG] checkpoint write_logit={wl.item():.4f} sigmoid={sig:.6f} -> alpha={0.01*sig:.8f}")
    pm.load_state_dict(ckpt["phase_memory"])
    adapt.load_state_dict(ckpt["retrieval_adapter"])
    adapt.eval(); pm.eval()
    print(f"  ✓ Retrieval adapter 加载成功: {checkpoint_path}")

    _hook_handle = [None]
    _hook_calls = [0]
    orig_prepare = model.prepare_condition

    def _patched_prepare(*args, **kwargs):
        result = orig_prepare(*args, **kwargs)
        if result[0] is not None and _hook_handle[0] is None:
            enc_hs = result[0]
            L_enc = enc_hs.shape[1]
            B_e, D_h = enc_hs.shape[0], enc_hs.shape[-1]

            parser = LyricsStructureParser()
            parsed = parser.parse(lyrics_text, num_chunks=L_enc)
            section_ids = parsed.section_type_ids
            units, _, debug = parse_lyrics_to_units(lyrics_text, section_ids, auto_transition_ratios={})
            sc = build_duration_scaffold(units, text_len=L_enc, tag_control_mask=debug.get("tag_control_mask"))
            sc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}

            U = len(sc["unit_boundaries"]) - 1
            c_all = (sc["unit_boundaries"][:-1] + sc["unit_boundaries"][1:]) / 2
            mu_all = sc["unit_duration"] / sc["unit_duration"].sum()
            usid = sc["unit_section_ids"]; uil = sc["lyric_unit_mask"]
            t2u = sc["token_to_unit"]; lym = sc["lyric_mask"]

            eh_f = enc_hs.float()
            uth_list = []
            for uid in range(U):
                is_l = uil[uid].item()
                tmask = (t2u == uid) & lym if is_l else (t2u == uid)
                tmb = tmask.unsqueeze(0).expand(B_e, -1)
                if tmb.any():
                    uth_list.append(eh_f[tmb].view(B_e, -1, D_h).mean(dim=1))
                else:
                    uth_list.append(torch.zeros(B_e, D_h, device=device, dtype=torch.float32))
            unit_text_hidden = torch.stack(uth_list, dim=1)
            p_audio = torch.linspace(0, 1, T_eff, device=device, dtype=torch.float32).unsqueeze(0)

            print(f"  Retrieval: U={U} units, L_enc={L_enc}, T_eff={T_eff}")

            def _hook(_mod, _inp, out):
                del _mod, _inp
                H = out[0]
                B, T_cur = H.shape[0], H.shape[1]
                pa = p_audio[:, :T_cur]
                if B > pa.shape[0]:
                    pa = pa.expand(B, -1).contiguous()
                with torch.no_grad():
                    ps = pm(H.float(), torch.zeros(B, device=device, dtype=torch.float32))
                    dh, Pi, diag = adapt(
                        hidden_states=H.float(), text_hidden=None, pm_state=ps,
                        p_audio=pa.float(), unit_text_hidden=unit_text_hidden.float(),
                        unit_c_pos=c_all.unsqueeze(0).float(), unit_mass=mu_all.unsqueeze(0).float(),
                        unit_section_id=usid.unsqueeze(0), unit_is_lyric=uil.unsqueeze(0),
                    )
                dh_rms = dh.pow(2).mean().sqrt().item()
                h_rms = H.pow(2).mean().sqrt().item()
                gate_m = diag.get('gate_mean', -1)
                gate_s = diag.get('gate_std', -1)
                wa = diag.get('write_alpha', -1)
                wr = diag.get('write_ratio', -1)
                if _hook_handle[0] is not None:
                    _hook_calls[0] += 1
                    if _hook_calls[0] <= 3 or _hook_calls[0] % 10 == 0:
                        print(f"    [HOOK #{_hook_calls[0]}] δh_rms={dh_rms:.4f} h_rms={h_rms:.1f} "
                              f"δh/h={dh_rms/max(h_rms,1e-8):.4f} "
                              f"gate={gate_m:.3f}±{gate_s:.3f} α={wa:.5f} w/r={wr:.5f}", flush=True)
                return ((H + dh.to(dtype=H.dtype)), *out[1:])

            _hook_handle[0] = model.decoder.layers[12].register_forward_hook(_hook)
            _hook_calls[0] = 0
        return result

    model.prepare_condition = _patched_prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="baseline",
                        choices=["baseline", "sinkhorn_only", "sinkhorn_pmgate"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--duration", type=int, default=268)
    args = parser.parse_args()

    if args.mode != "baseline" and not args.checkpoint:
        args.checkpoint = BEST_CHECKPOINT
        print(f"  使用默认 checkpoint: {args.checkpoint}")

    print("=" * 70)
    print("ACE-Step 1.5 文本到音乐生成" + (f" — Mode: {args.mode}" if args.mode != "baseline" else ""))
    print("=" * 70)

    # ========== 1. 初始化处理器 ==========
    print("\n[1/4] 初始化处理器...")
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    # ========== 2. 初始化 DiT 服务 ==========
    print("\n[2/4] 初始化 DiT 模型...")
    dit_status, dit_success = dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )

    if not dit_success:
        print(f"❌ DiT 初始化失败: {dit_status}")
        sys.exit(1)
    print(f"✓ DiT 初始化成功")
    print(dit_status)

    model = dit_handler.model
    device = next(model.parameters()).device
    model.config.use_section_rope_offset = False
    for layer in model.decoder.layers:
        if getattr(layer, "use_section_rope", False): layer.use_section_rope = False
        if getattr(layer, "use_phase_memory", False): layer.use_phase_memory = False

    # ---- Retrieval hook（仅非 baseline 模式）----
    if args.mode != "baseline":
        print("\n 加载 Retrieval Adapter...")
        # 歌词文本必须包含换行，否则 parse_lyrics_to_units 无法正确解析
        lyrics_text = """[INTRO]

[VERSE]
In the quiet of the night
I can hear my heartbeat slow
Memories fading like the light
Letting everything I know go

[CHORUS]
Underneath the sky so wide
I will find my way back home
Nothing left for me to hide
I am never alone

[INSTRUMENTAL]

[VERSE]
Every step I take is new
Every breath a brand new start
All the things I thought I knew
Fade away into the dark

[CHORUS]
Underneath the sky so wide
I will find my way back home
Nothing left for me to hide
I am never alone

[OUTRO]
"""
        _inject_retrieval(
            model,
            lyrics_text=lyrics_text,
            T_eff=int(args.duration * 25),
            checkpoint_path=args.checkpoint,
            use_pm_gate=(args.mode == "sinkhorn_pmgate"),
            device=device,
        )

    # ========== 3/4. 初始化 5Hz LM 服务 ==========
    print("\n[3/4] 初始化 5Hz 语言模型...")
    llm_success = llm_handler.initialize(
        checkpoint_dir=str(MODEL_ROOT),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )

    if not llm_success:
        print("❌ 5Hz LM 初始化失败")
        sys.exit(1)
    print("✓ 5Hz LM 初始化成功")

    # ========== 4/4. 配置生成参数 ==========
    print("\n[4/4] 配置生成参数...")

    params = GenerationParams(
        task_type="text2music",
        caption="pop, female vocal, piano, guitar, drums, emotional, atmospheric, 120 bpm, C major",
        lyrics="""[INTRO]

[VERSE]
In the quiet of the night
I can hear my heartbeat slow
Memories fading like the light
Letting everything I know go

[CHORUS]
Underneath the sky so wide
I will find my way back home
Nothing left for me to hide
I am never alone

[INSTRUMENTAL]

[VERSE]
Every step I take is new
Every breath a brand new start
All the things I thought I knew
Fade away into the dark

[CHORUS]
Underneath the sky so wide
I will find my way back home
Nothing left for me to hide
I am never alone

[OUTRO]""",
        instrumental=False,
        bpm=137,
        keyscale="E major",
        timesignature="4",
        vocal_language="zh",
        duration=args.duration,
        inference_steps=50,
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
        seeds=[args.seed],
    )

    # ========== 5. 生成音乐 ==========
    print("\n" + "=" * 70)
    print("开始生成音乐...")
    print("=" * 70)

    output_dir = Path(args.output_dir) if args.output_dir else ACE_STEP_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=str(output_dir),
    )

    # ========== 6. 处理结果 ==========
    print("\n" + "=" * 70)
    if result.success:
        print("✓ 音乐生成成功!")
        print("=" * 70)
        for i, audio in enumerate(result.audios):
            print(f"\n生成结果 {i+1}:")
            print(f"  文件路径: {audio['path']}")
            print(f"  种子: {audio['params']['seed']}")
            print(f"  时长: {audio['params'].get('duration', 'N/A')} 秒")
    else:
        print(f"❌ 生成失败: {result.error}")
        sys.exit(1)

    print(f"\n输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
