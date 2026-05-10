"""Clean structured lyrics with Bailian LLM API.

This script reads a JSON dataset, fixes likely section-tag position mistakes
in structured lyrics, and removes noisy lines such as credits/ads.
It uses Bailian OpenAI-compatible chat API.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# Per user request, values are embedded in code for convenience.
BAILIAN_API_KEY = "sk-d06075dbd6914fa797f9bd3787a96e34"
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BAILIAN_MODEL = "glm-5.1"

DEFAULT_INPUT_PATH = "/root/autodl-tmp/musicdata/results.jsonl"
DEFAULT_OUTPUT_PATH = "/root/autodl-tmp/musicdata/results.cleaned.jsonl"

LYRICS_CANDIDATE_FIELDS = [
    "lyric_text",
    "structured_lyrics",
    "lyrics_structured",
    "lyrics",
    "lyric",
    "lrc",
    "text",
]

NOISE_PATTERNS = [
    r"(?i)作词|作曲|编曲|制作人|监制|录音|混音|母带|出品",
    r"(?i)词\s*[:：]|曲\s*[:：]|编\s*[:：]",
    r"(?i)copyright|all rights reserved|publisher|publishing",
    r"(?i)关注|扫码|公众号|微博|抖音|快手|小红书|B站",
    r"(?i)subscribe|follow|like\s+and\s+share|download",
    r"(?i)QQ\s*群|vx\s*[:：]|微信\s*[:：]|联系",
    r"(?i)广告|赞助|商务合作|推广",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Clean structured lyrics with Bailian API")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input JSON path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path")
    parser.add_argument(
        "--lyrics-field",
        default="",
        help="Lyrics field name. Empty means auto-detect from common fields.",
    )
    parser.add_argument(
        "--write-field",
        default="",
        help="Field to write cleaned lyrics. Empty means overwrite detected field.",
    )
    parser.add_argument("--model", default=BAILIAN_MODEL, help="Bailian model name")
    parser.add_argument("--api-key", default=BAILIAN_API_KEY, help="Bailian API key")
    parser.add_argument("--base-url", default=BAILIAN_BASE_URL, help="Bailian base URL")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Only process first N items. 0 means all.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save checkpoint every N items.",
    )
    return parser.parse_args()


def strip_noise_lines(text: str) -> Tuple[str, List[str]]:
    """Remove obvious credit/ad lines before LLM cleanup.

    Returns cleaned text and removed lines.
    """
    if not text.strip():
        return text, []

    removed: List[str] = []
    kept: List[str] = []

    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            kept.append(line)
            continue

        matched = any(re.search(pat, line_stripped) for pat in NOISE_PATTERNS)
        if matched:
            removed.append(line)
        else:
            kept.append(line)

    return "\n".join(kept).strip(), removed


def find_lyrics_field(record: Dict[str, Any], preferred_field: str = "") -> Optional[str]:
    """Find lyrics field from preferred name or common candidates."""
    if preferred_field and preferred_field in record:
        return preferred_field

    for field in LYRICS_CANDIDATE_FIELDS:
        if field in record and isinstance(record[field], (str, dict, list)):
            return field
    return None


def to_prompt_text(value: Any) -> str:
    """Convert lyrics payload to text for LLM input."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("Model output does not contain JSON object")

    depth = 0
    in_string = False
    escape = False
    end = -1

    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        raise ValueError("Unclosed JSON object in model output")

    return json.loads(text[start:end])


def call_bailian_cleanup(
    lyrics_text: str,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
) -> Dict[str, Any]:
    """Call Bailian chat API and return parsed cleanup result."""
    system_prompt = (
        "你是歌词清洗与结构修复助手。"
        "你的任务是修复结构化歌词标签位置错误，并移除与歌词无关的冗余内容。"
        "例如：乐队成员信息、作词作曲编曲、版权声明、联系方式、广告、推广文案。"
        "必须尽量保持原歌词语义与顺序，不要凭空扩写。"
        "如果输入已经很干净，尽量少改动。"
        "只输出 JSON，不要输出 markdown 代码块。"
    )
    user_prompt = (
        "请清洗下面的结构化歌词，并修复标签位置。\n\n"
        "输出 JSON 格式严格为：\n"
        "{\n"
        "  \"cleaned_lyrics\": \"<清洗后歌词，保留结构标签>\",\n"
        "  \"removed_lines\": [\"<被删除的典型行，可为空数组>\"],\n"
        "  \"change_notes\": \"<简短说明>\"\n"
        "}\n\n"
        "输入歌词：\n"
        f"{lyrics_text}"
    )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Bailian API error {response.status_code}: {response.text[:500]}")

    body = response.json()
    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if not content:
        raise RuntimeError("Empty content from Bailian API")

    result = extract_json_object(content)
    if "cleaned_lyrics" not in result:
        raise ValueError("Model JSON missing 'cleaned_lyrics'")
    return result


def load_dataset(path: Path) -> Tuple[List[Dict[str, Any]], Any, Optional[str]]:
    """Load dataset and return records plus original root wrapper info."""
    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
        return records, {"_format": "jsonl"}, "__jsonl__"

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw, raw, None

    if isinstance(raw, dict):
        for key in ("data", "results", "items"):
            if key in raw and isinstance(raw[key], list):
                return raw[key], raw, key

    raise ValueError("Unsupported JSON format: expected list or dict with data/results/items list")


def save_dataset(
    output_path: Path,
    root_obj: Any,
    root_list_key: Optional[str],
    records: List[Dict[str, Any]],
) -> None:
    """Save records back to output JSON while preserving root shape."""
    if root_list_key == "__jsonl__":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    if root_list_key is None:
        out_obj = records
    else:
        out_obj = dict(root_obj)
        out_obj[root_list_key] = records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)


def process_records(args: argparse.Namespace) -> None:
    """Process all records and write cleaned dataset."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records, root_obj, root_list_key = load_dataset(input_path)
    total = len(records)
    limit = args.max_items if args.max_items > 0 else total
    limit = min(limit, total)

    print(f"Loaded {total} records, processing {limit} records")

    success = 0
    skipped = 0
    failed = 0

    for idx in range(limit):
        record = records[idx]
        if not isinstance(record, dict):
            skipped += 1
            continue

        field = find_lyrics_field(record, args.lyrics_field)
        if not field:
            skipped += 1
            continue

        write_field = args.write_field or field
        raw_lyrics = to_prompt_text(record[field])
        rule_cleaned, removed_by_rules = strip_noise_lines(raw_lyrics)

        if not rule_cleaned.strip():
            record[write_field] = ""
            record["lyrics_clean_meta"] = {
                "status": "rule_only",
                "removed_lines": removed_by_rules,
                "change_notes": "Removed by local rules; no content left.",
            }
            success += 1
            continue

        try:
            llm_result = call_bailian_cleanup(
                lyrics_text=rule_cleaned,
                api_key=args.api_key,
                model=args.model,
                base_url=args.base_url,
                timeout=args.timeout,
            )
            cleaned_lyrics = llm_result.get("cleaned_lyrics", "").strip() or rule_cleaned
            record[write_field] = cleaned_lyrics
            record["lyrics_clean_meta"] = {
                "status": "ok",
                "removed_lines": removed_by_rules + llm_result.get("removed_lines", []),
                "change_notes": llm_result.get("change_notes", ""),
                "source_field": field,
                "target_field": write_field,
            }
            success += 1
        except Exception as exc:
            record[write_field] = rule_cleaned
            record["lyrics_clean_meta"] = {
                "status": "error",
                "error": str(exc),
                "removed_lines": removed_by_rules,
                "change_notes": "LLM failed; fallback to rule-based cleaned lyrics.",
                "source_field": field,
                "target_field": write_field,
            }
            failed += 1

        if (idx + 1) % max(args.save_every, 1) == 0:
            save_dataset(output_path, root_obj, root_list_key, records)
            print(f"Checkpoint saved at {idx + 1}/{limit}")

    save_dataset(output_path, root_obj, root_list_key, records)
    print(
        "Done. "
        f"success={success}, skipped={skipped}, failed={failed}, output={output_path}"
    )
    if success == 0 and failed == 0 and skipped > 0:
        print(
            "Hint: all records were skipped because no lyrics field was detected. "
            "Try --lyrics-field lyric_text or update LYRICS_CANDIDATE_FIELDS."
        )


def main() -> None:
    """CLI entry."""
    args = parse_args()
    process_records(args)


if __name__ == "__main__":
    main()
