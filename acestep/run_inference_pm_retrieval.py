#!/usr/bin/env python3
"""
ACE-Step 1.5 + PM-Retrieval 推理脚本

加载训练好的 PMRetrievalPhaseMemory + LyricRetrievalAdapter checkpoint，
在 layer 12 注入 retrieval residual，然后走标准生成流程。
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
from acestep.phase_memory import PMRetrievalPhaseMemory, LyricRetrievalAdapter


def load_pm_retrieval(model, ckpt_path, device):
    D = model.config.hidden_size
    pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True)
    adapt = LyricRetrievalAdapter(
        hidden_dim=D, text_dim=D, pm_dim=256, d_r=64,
        residual_scale=0.1, gamma_init=0.01,
        use_adapter_scaffold_prior=False,
    )

    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    pm_sd = saved.get("phase_memory", saved)
    adapt_sd = saved.get("retrieval_adapter", saved)
    pm.load_state_dict(pm_sd, strict=False)
    adapt.load_state_dict(adapt_sd, strict=False)

    pm = pm.to(device).float().eval()
    adapt = adapt.to(device).float().eval()

    print(f"[PM-Retrieval] PM: {len(pm_sd)} tensors, Adapter: {len(adapt_sd)} tensors")
    print(f"[PM-Retrieval] gamma_r = {adapt.gamma_r.item():.5f}")

    model.add_module("pm_retrieval_pm", pm)
    model.add_module("pm_retrieval_adapter", adapt)

    enc_cache = {}

    def capture_enc_hook(module, inputs, kwargs):
        eh = kwargs.get("encoder_hidden_states", None)
        if eh is not None:
            enc_cache["value"] = eh.float()

    pre_handle = model.decoder.register_forward_pre_hook(capture_enc_hook, with_kwargs=True)

    def inject_hook(module, inputs, output):
        h = output[0].float()
        B, T = h.shape[:2]
        dev = h.device

        text_hidden = enc_cache.get("value")
        if text_hidden is None or text_hidden.shape[0] != B:
            text_hidden = h

        T_t = text_hidden.shape[1]
        p_audio = torch.linspace(0, 1, T, device=dev, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        c_text = torch.linspace(0, 1, T_t, device=dev, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        section_id = torch.zeros(B, T_t, dtype=torch.long, device=dev)
        token_type_id = torch.zeros(B, T_t, dtype=torch.long, device=dev)
        t_emb = torch.zeros(B, 128, device=dev, dtype=torch.float32)

        with torch.no_grad():
            pm_state = pm(h, None)
            ret_res, _, _ = adapt(
                hidden_states=h, text_hidden=text_hidden, pm_state=pm_state,
                p_audio=p_audio, c_text=c_text, section_id=section_id,
                token_type_id=token_type_id, timestep_emb=t_emb,
                use_scaffold_prior=True,
            )
            h_new = h + adapt.gamma_r * ret_res

        return (h_new.to(dtype=output[0].dtype), *output[1:])

    hook_handle = model.decoder.layers[12].register_forward_hook(inject_hook)
    print("[PM-Retrieval] Layer 12 hook installed")
    return pre_handle, hook_handle


def main():
    print("=" * 70)
    print("ACE-Step 1.5 + PM-Retrieval 文本到音乐生成")
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

    model = dit_handler.model.eval()

    # ========== 2.5. 加载 PM-Retrieval 权重 ==========
    print("\n[2.5/4] 加载 PM-Retrieval 权重...")
    ckpt_path = "/root/autodl-tmp/pm_retrieval_1epoch_v3/checkpoints/epoch_2_loss_1.3316/pm_retrieval.pt"
    if os.path.exists(ckpt_path):
        pre_handle, hook_handle = load_pm_retrieval(model, ckpt_path, model.device)
        print(f"✓ PM-Retrieval 权重加载成功: {ckpt_path}")
    else:
        print(f"ℹ️ PM-Retrieval 权重未找到 ({ckpt_path})，跳过")
        pre_handle, hook_handle = None, None

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

        caption="ballad, pop, female vocal, piano, strings, drums, romantic, melancholic, 137 bpm, E major",
        lyrics="""[Verse] 爱总忽然退潮 心慌乱触礁 沉没在深海里 看海面闪耀 但回忆像水草 紧紧的缠绕 梦才温热眼角 就冰冷掉 努力越过风暴 向着未来飘 我们才会遇到 感动的拥抱 你总是能知道 我的坚强剩多少 [Pre-chorus] 给我最刚好的依靠 [Chorus] 你手心的太阳 只轻放在我背上 委屈就能笑着落泪 被释放 在手心的太阳 黑暗里特别明亮 让远路好像 是一种分享 而不是漫长 [Inst] 让眼睛看不到 嫉妒的燃烧 [Verse] 让耳朵听不到 谎言的吵闹 再没有人相信 爱能永恒那一秒 我们正坚定的微笑 [Chorus] 你手心的太阳 有种安定的力量 就算世界再乱我也 不心慌 我手心的太阳 或许只像个月亮 却用所有爱 为你投射我最暖的光芒 你手心的太阳 只轻放在我背上 [Bridge] 委屈就能笑着落泪 被释放 在手心的太阳 黑暗里特别明亮 让远路好像 是一种分享 你手心的太阳 有种安定的力量 [Chorus] 就算世界再乱我也 不心慌 我手心的太阳 或许只像个月亮 却用所有爱 为你投射我 最暖的光芒""",
        instrumental=False,

        bpm=137,
        keyscale="E major",
        timesignature="4",
        vocal_language="zh",
        duration=-1,

        inference_steps=50,
        guidance_scale=7.0,
        seed=42,

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

    # ========== 5. 生成音乐 ==========
    print("\n" + "=" * 70)
    print("开始生成音乐...")
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
    print("[PM-Retrieval] Handles removed")

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
