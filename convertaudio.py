from pathlib import Path

# ================= 配置区 =================
# 你的音频所在的文件夹路径
WORK_DIR = Path("/root/autodl-tmp/musicdata/audios") 
# ==========================================

def main():
    print(f"开始扫描目录: {WORK_DIR}")
    
    # 查找所有的 .audio 文件
    audio_files = list(WORK_DIR.glob("*.audio"))
    
    if not audio_files:
        print("❌ 没找到任何 .audio 文件，可能已经转换过了？")
        return

    print(f"共发现 {len(audio_files)} 个音频文件，开始原地修改后缀...")
    
    count = 0
    for src_file in audio_files:
        # src_file.with_suffix(".mp3") 会直接把路径的最后一部分替换为 .mp3
        tgt_file = src_file.with_suffix(".mp3")
        
        # 原地重命名 (瞬间完成，0 空间损耗)
        src_file.rename(tgt_file)
        count += 1

    print("\n" + "="*40)
    print(f"🎉 音频后缀修改完毕！")
    print(f"成功将 {count} 个文件变成了标准的 .mp3 格式。")
    print("="*40)

if __name__ == "__main__":
    main()