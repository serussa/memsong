import json
from pathlib import Path

# ================= 配置区 =================
CAPTIONS_JSONL = Path("/root/autodl-tmp/musicdata/dataset_tags_repaired.jsonl")      # 替换为你的 caption jsonl 文件路径
LYRICS_JSONL= Path("/root/autodl-tmp/musicdata/results_cleaned_final.jsonl")# 替换为你的歌词 jsonl 文件路径
OUTPUT_DIR = Path("/root/autodl-tmp/musicdata/dataset")            # 文本生成的输出文件夹
# ==========================================

def main():
    # 创建输出文件夹
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 建立歌词检索库 (Hash Map)
    print("正在构建歌词索引...")
    lyrics_map = {}
    with open(LYRICS_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            data = json.loads(line)
            
            # 原始路径可能是 Windows 格式 "F:\\...\\xxx.audio"，统一替换斜杠并提取文件名
            raw_path = data.get("audio_path", "")
            filename = Path(raw_path.replace("\\", "/")).name 
            
            lyrics_map[filename] = data.get("lyric_text", "")
            
    print(f"构建完成，共在内存中缓存了 {len(lyrics_map)} 条歌词。")

    # 2. 以 Caption 为基准进行遍历匹配
    print("正在匹配数据并生成文本文件...")
    match_count = 0
    missing_lyric_count = 0

    with open(CAPTIONS_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            data = json.loads(line)

            # 提取 meta 信息获取真正的文件名
            meta = data.get("caption_tag_fix_meta", {})
            filename = meta.get("audio_ref")

            if not filename or filename not in data:
                continue

            caption_text = data[filename]

            # 寻找是否有对应的歌词
            if filename not in lyrics_map:
                missing_lyric_count += 1
                continue
            
            lyric_text = lyrics_map[filename]

            # 3. 匹配成功，开始写入目标文件夹
            # Path(filename).stem 会去掉 ".audio" 后缀，提取出纯文件名 (例如 "00249..._1686")
            stem = Path(filename).stem
            
            tgt_lyric_file = OUTPUT_DIR / f"{stem}.lyrics.txt"
            tgt_caption_file = OUTPUT_DIR / f"{stem}.caption.txt"

            # 写入文本 (指定 utf-8 编码防止中文乱码)
            tgt_caption_file.write_text(caption_text, encoding='utf-8')
            tgt_lyric_file.write_text(lyric_text, encoding='utf-8')

            match_count += 1

    # 4. 打印处理报告
    print("\n" + "="*40)
    print(f"🎉 文本处理完成！")
    print(f"成功匹配并生成了: {match_count} 组 txt 对 (共 {match_count * 2} 个文件)")
    print(f"因匹配不到歌词而跳过: {missing_lyric_count} 个")
    print("="*40)

if __name__ == "__main__":
    main()