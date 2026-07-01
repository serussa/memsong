#!/usr/bin/env python3
"""
ACE-Step 1.5 文本到音乐生成 + PhaseMemory 推理
使用最新训练的 PhaseMemory 权重
"""

import os
import sys
from pathlib import Path

# 设置项目根目录
ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
CHECKPOINT_DIR = Path("/root/autodl-tmp/lyrics_checkpoints/checkpoints")

# 自动选择最新 checkpoint（按 epoch 编号数值排序）
checkpoints = sorted(CHECKPOINT_DIR.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
if not checkpoints:
    print(f"❌ 未找到 checkpoint，路径: {CHECKPOINT_DIR}")
    sys.exit(1)
LATEST_CKPT = checkpoints[-1]
print(f"使用 checkpoint: {LATEST_CKPT.name}")

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(ACE_STEP_ROOT))

# 强制离线模式（禁止自动下载模型）
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.training.phase_memory_checkpoint import load_phase_memory_weights


def main():
    print("=" * 70)
    print(f"ACE-Step 1.5 文本到音乐生成 (PhaseMemory)")
    print(f"Checkpoint: {LATEST_CKPT}")
    print("=" * 70)

    # ========== 1. 初始化处理器 ==========
    print("\n[1/4] 初始化处理器...")
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    # ========== 2. 初始化 DiT 模型 ==========
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

    # ========== 加载 PhaseMemory 权重 ==========
    print(f"\n加载 PhaseMemory 权重...")
    load_phase_memory_weights(
        dit_handler.model,
        str(LATEST_CKPT),
    )
    print(f"✓ PhaseMemory 权重加载成功")

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
        duration=280,
        inference_steps=50,
        guidance_scale=10.0,
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
