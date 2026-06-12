#!/usr/bin/env python3
"""
ACE-Step 1.5 文本到音乐生成脚本
使用本地下载的模型
"""

import os
import sys
from pathlib import Path

# 设置项目根目录
ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(ACE_STEP_ROOT))

# 强制离线模式（禁止自动下载模型）
os.environ["ACESTEP_OFFLINE"] = "1"
# 仅校验所需组件（vae + embedding + 当前 DiT）
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music


def main():
    print("=" * 70)
    print("ACE-Step 1.5 文本到音乐生成")
    print("=" * 70)
    
    # ========== 1. 初始化处理器 ==========
    print("\n[1/4] 初始化处理器...")
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()
    
    # ========== 2. 初始化 DiT 服务 ==========
    print("\n[2/4] 初始化 DiT 模型...")
    # 直接使用本地下载的模型目录作为 project_root
    # 这样 initialize_service 会使用 MODEL_ROOT 作为 checkpoint_path
    dit_status, dit_success = dit_handler.initialize_service(
        project_root=str(MODEL_ROOT),  # 使用本地下载的模型目录
        config_path="acestep-v15-sft",
        device="cuda",  # 或 "cpu", "auto"
        use_flash_attention=False,  # aarch64 架构设为 False
        compile_model=False,  # 首次运行建议设为 False
        offload_to_cpu=False,  # GPU 内存不足时设为 True
    )
    
    if not dit_success:
        print(f"❌ DiT 初始化失败: {dit_status}")
        sys.exit(1)
    print(f"✓ DiT 初始化成功")
    print(dit_status)
    
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
    
    # 文本到音乐生成参数
    params = GenerationParams(
        # 任务类型
        task_type="text2music",
        
        # 文本输入
        caption="ballad, pop, vocal, live recording, piano, rock band, drums, nostalgic, emotional, inspiring",
        lyrics="""[Intro: Crowd Cheers & Piano] [Verse 1: Vocal] 你站在讲台上 把难题说得亮堂 一支笔一块黑板 照见我眼里的慌 你总把话说实在 像冬天一杯热汤 不绕弯不撞墙 把路给我摆在前方 [Pre-Chorus: Vocal] 那时候我害怕 怕未来太长 你一句别怂 我就敢往前闯 [Chorus: Vocal] 张雪峰老师 我还记得你 一句一句 把我拉出迷雾里 张雪峰老师 我还记得你 那些年少的愁 你替我扛起 [Instrumental Break: Pop Rock Band] [Verse 2: Vocal] 粉笔灰落在肩上 像雪一样安静 你笑着说前途远 别先把自个儿看轻 多少个深夜回家 我还在翻那页纸 你说过的每句话 都在我心里理清 [Pre-Chorus: Vocal] 后来风吹过来 我也学会了 不躲不退 往自己的山坡走 [Chorus: Vocal] 张雪峰老师 我还记得你 一句一句 把我拉出迷雾里 张雪峰老师 我还记得你 那些年少的愁 你替我扛起 [Bridge: Vocal] 如果有一天 我走得很远 也会想起那间教室的光线 想起你拍着桌子 说别怕吃苦 说这条路 总会有人走出去 [Chorus: Vocal] 张雪峰老师 我还记得你 一句一句 把我拉出迷雾里 张雪峰老师 我还记得你 那些年少的愁 你替我扛起 [Outro: Vocal] 张雪峰老师 我还记得你 风再大 我也会继续前进 张雪峰老师 我还记得你 你给过的勇气 我一直放在心里 [Music fades out]
""",
        instrumental=False,  # False 表示有歌词
        
        # 音乐元数据（可选）
        bpm=137,
        keyscale="",
        timesignature="",  # 4/4 拍
        vocal_language="zh",  # 语言: 中文
        duration=268,  # 3分钟的标准流行歌曲长度
        
        # 生成参数
        inference_steps=50,  # base 通常可用更多步数提升质量
        guidance_scale=7.0,  # CFG 强度
        seed=42,  # 随机种子
        
        # 启用 LM 规划
        thinking=True,
        use_cot_metas=True,
        use_cot_caption=True,
        lm_temperature=0.75,
    )
    
    # 配置生成设置
    config = GenerationConfig(
        batch_size=1,  # 生成数量
        audio_format="flac",  # 输出格式
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