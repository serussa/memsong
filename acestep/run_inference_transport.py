#!/usr/bin/env python3
"""
ACE-Step 1.5 + TransportRetrievalAdapter 推理脚本

加载训练好的 TransportRetrievalAdapter + PMRetrievalPhaseMemory checkpoint，
注入 layer 12 的 unit-level Sinkhorn transport retrieval hidden residual，
然后走标准生成流程。
"""

import os
import sys
from pathlib import Path

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

sys.path.insert(0, str(ACE_STEP_ROOT))

os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

import torch

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.phase_memory import (
    PMRetrievalPhaseMemory, TransportRetrievalAdapter,
    parse_lyrics_to_units, build_duration_scaffold,
)


def load_transport_adapter(model, ckpt_path, device):
    """Load TransportRetrievalAdapter checkpoint and install layer-12 hooks."""
    D = model.config.hidden_size

    pm = PMRetrievalPhaseMemory(
        dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True,
    )
    adapt = TransportRetrievalAdapter(
        hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        sinkhorn_iters=5, transport_sigma=0.18, transport_qk_scale=1.0,
        write_alpha_init=1e-4, write_alpha_max=1e-3,
    )

    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    pm_sd = saved.get("phase_memory", saved)
    adapt_sd = saved.get("retrieval_adapter", saved)
    pm.load_state_dict(pm_sd, strict=False)
    adapt.load_state_dict(adapt_sd, strict=False)

    pm = pm.to(device).float().eval()
    adapt = adapt.to(device).float().eval()

    print(f"[Transport] PM: {len(pm_sd)} tensors, Adapter: {len(adapt_sd)} tensors")
    print(f"[Transport] write_alpha = {adapt.write_alpha.item():.6f}")

    model.add_module("transport_pm", pm)
    model.add_module("transport_adapter", adapt)

    # Persistent cache built once in pre-hook from real encoder_hidden_states
    transport_cache: dict = {}

    # ---- Pre-hook: capture encoder_hidden_states + build scaffold ----
    def capture_enc_hook(module, inputs, kwargs):
        eh = kwargs.get("encoder_hidden_states", None)
        if eh is None:
            return
        L = eh.shape[1]
        transport_cache["text_hidden"] = eh.float()

        # Build scaffold once from real text encoder output
        if "scaffold_ready" not in transport_cache:
            meta = getattr(model, "_gen_metadata", {})
            lyrics_text = meta.get("lyrics", "") if isinstance(meta, dict) else ""
            if lyrics_text and L > 0:
                fake_section_ids = torch.zeros(L, dtype=torch.long, device="cpu")
                units, _, debug = parse_lyrics_to_units(lyrics_text, fake_section_ids)
                tcm = debug.get("tag_control_mask", None)
                sc = build_duration_scaffold(units, text_len=L, tag_control_mask=tcm)
                lyric_uids = torch.where(sc["lyric_unit_mask"])[0]
                K = len(lyric_uids)
                transport_cache["K"] = K
                if K > 0:
                    c_all = (sc["unit_boundaries"][:-1] + sc["unit_boundaries"][1:]) / 2
                    mu_all = sc["unit_duration"]
                    c_unit = c_all[lyric_uids]
                    mu = mu_all[lyric_uids]
                    usid = sc["unit_section_ids"][lyric_uids]

                    # Pool text_hidden per lyric unit (cast to float32)
                    eh_f = eh.float()
                    token_to_unit = sc["token_to_unit"].to(eh.device)
                    lyric_mask = sc["lyric_mask"].to(eh.device)
                    B = eh.shape[0]
                    unit_h_list = []
                    for uid in lyric_uids.tolist():
                        token_mask = (token_to_unit == uid) & lyric_mask
                        if token_mask.any():
                            pooled = eh_f[:, token_mask].mean(dim=1)
                        else:
                            pooled = torch.zeros(B, eh_f.shape[-1], device=eh.device, dtype=torch.float32)
                        unit_h_list.append(pooled)
                    unit_text_hidden = torch.stack(unit_h_list, dim=1)

                    transport_cache["unit_text_hidden"] = unit_text_hidden
                    transport_cache["c_unit"] = c_unit.to(eh.device)
                    transport_cache["mu"] = mu.to(eh.device)
                    transport_cache["usid"] = usid.to(eh.device)
                    print(f"[Transport] Scaffold built: K={K} lyric units from L={L} text tokens")
                else:
                    print(f"[Transport] WARNING: K=0 lyric units for L={L} tokens")
            transport_cache["scaffold_ready"] = True

    pre_handle = model.decoder.register_forward_pre_hook(capture_enc_hook, with_kwargs=True)

    # ---- Inject hook: transport retrieval at layer 12 ----
    def inject_hook(module, inputs, output):
        h = output[0].float()
        B, T = h.shape[:2]
        dev = h.device

        K = transport_cache.get("K", 0)
        if K > 0:
            unit_text_hidden = transport_cache["unit_text_hidden"].to(dev)
            c_unit_b = transport_cache["c_unit"].unsqueeze(0).expand(B, -1)
            mu_b = transport_cache["mu"].unsqueeze(0).expand(B, -1)
            usid_b = transport_cache["usid"].unsqueeze(0).expand(B, -1)
            p_audio = torch.linspace(0, 1, T, device=dev, dtype=torch.float32).unsqueeze(0).expand(B, -1)

            with torch.no_grad():
                pm_state = pm(h, None)
                delta_h, _, _ = adapt(
                    hidden_states=h,
                    text_hidden=unit_text_hidden,
                    pm_state=pm_state,
                    p_audio=p_audio,
                    unit_text_hidden=unit_text_hidden,
                    unit_c_pos=c_unit_b,
                    unit_mass=mu_b,
                    unit_section_id=usid_b,
                )
                h_new = h + delta_h
        else:
            h_new = h

        return (h_new.to(dtype=output[0].dtype), *output[1:])

    hook_handle = model.decoder.layers[12].register_forward_hook(inject_hook)
    print("[Transport] Layer 12 Sinkhorn transport hook installed")
    return pre_handle, hook_handle, transport_cache


def main():
    print("=" * 70)
    print("ACE-Step 1.5 + TransportRetrievalAdapter (Sinkhorn) 推理")
    print("=" * 70)

    # ========== 1. Initialize handlers ==========
    print("\n[1/4] Initializing handlers...")
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    # ========== 2. Initialize DiT ==========
    print("\n[2/4] Initializing DiT model...")
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
    print(f"[OK] DiT initialized")
    print(dit_status)

    model = dit_handler.model.eval()

    # ========== 2.5. Load Transport checkpoint ==========
    print("\n[2.5/4] Loading TransportRetrievalAdapter checkpoint...")
    ckpt_path = "/tmp/transport_1epoch_dr256/final/pm_retrieval.pt"
    if os.path.exists(ckpt_path):
        pre_handle, hook_handle, transport_cache = load_transport_adapter(
            model, ckpt_path, model.device,
        )
        print(f"[OK] Transport checkpoint loaded: {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found: {ckpt_path}")
        pre_handle, hook_handle = None, None

    # ========== 3. Initialize 5Hz LM ==========
    print("\n[3/4] Initializing 5Hz language model...")
    llm_success = llm_handler.initialize(
        checkpoint_dir=str(MODEL_ROOT),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
    )

    if not llm_success:
        print("[FAIL] 5Hz LM init failed")
        sys.exit(1)
    print("[OK] 5Hz LM initialized")

    # ========== 4. Configure generation ==========
    print("\n[4/4] Configuring generation...")

    lyrics = """[INTRO]

[VERSE]
爱总忽然退潮 心慌乱触礁
沉没在深海里 看海面闪耀
但回忆像水草 紧紧的缠绕
梦才温热眼角 就冰冷掉
努力越过风暴 向着未来飘
我们才会遇到 感动的拥抱
你总是能知道 我的坚强剩多少

[PRECHORUS]
给我最刚好的依靠

[CHORUS]
你手心的太阳 只轻放在我背上
委屈就能笑着落泪 被释放
在手心的太阳 黑暗里特别明亮
让远路好像 是一种分享 而不是漫长



[VERSE]
让眼睛看不到 嫉妒的燃烧
让耳朵听不到 谎言的吵闹
再没有人相信 爱能永恒那一秒
我们正坚定的微笑

[CHORUS]
你手心的太阳 有种安定的力量
就算世界再乱我也 不心慌
我手心的太阳 或许只像个月亮
却用所有爱 为你投射我最暖的光芒
你手心的太阳 只轻放在我背上



[BRIDGE]
委屈就能笑着落泪 被释放
在手心的太阳 黑暗里特别明亮
让远路好像 是一种分享
你手心的太阳 有种安定的力量

[CHORUS]
就算世界再乱我也 不心慌
我手心的太阳 或许只像个月亮
却用所有爱 为你投射我 最暖的光芒

[OUTRO]
"""

    # Store lyrics in model so the hook can access them
    model._gen_metadata = {"lyrics": lyrics}

    params = GenerationParams(
        task_type="text2music",
        caption="ballad, pop, female vocal, piano, strings, drums, romantic, melancholic, 137 bpm, E major",
        lyrics=lyrics,
        instrumental=False,
        bpm=137,
        keyscale="E major",
        timesignature="4",
        vocal_language="zh",
        duration=268,
        inference_steps=50,
        guidance_scale=7.0,
        seed=624,
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

    # ========== 5. Generate ==========
    print("\n" + "=" * 70)
    print("Generating music...")
    print("=" * 70)

    output_dir = ACE_STEP_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=str(output_dir),
    )

    if pre_handle is not None:
        pre_handle.remove()
    if hook_handle is not None:
        hook_handle.remove()
    print("[Transport] Hooks removed")

    # ========== 6. Results ==========
    print("\n" + "=" * 70)
    if result.success:
        print("[OK] Generation success!")
        print("=" * 70)
        for i, audio in enumerate(result.audios):
            print(f"\nResult {i+1}:")
            print(f"  Path: {audio['path']}")
            print(f"  Seed: {audio['params']['seed']}")
            print(f"  Duration: {audio['params'].get('duration', 'N/A')}s")
    else:
        print(f"[FAIL] Generation failed: {result.error}")
        sys.exit(1)

    print(f"\nOutput: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
