"""Clean structured lyrics with Bailian LLM API.

This script reads a JSON dataset, fixes likely section-tag position mistakes
in structured lyrics, and removes noisy lines such as credits/ads.
It uses Bailian OpenAI-compatible chat API.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# Per user request, values are embedded in code for convenience.
BAILIAN_API_KEY = "sk-d06075dbd6914fa797f9bd3787a96e34"
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BAILIAN_MODEL = "qwen3-max-2026-01-23"

DEFAULT_INPUT_PATH = "/root/autodl-tmp/musicdata/results.jsonl"
DEFAULT_OUTPUT_PATH = "/root/autodl-tmp/musicdata/results.cleaned.jsonl"
DEFAULT_LYRICS_FIELD = "lyric_text"

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
        default=DEFAULT_LYRICS_FIELD,
        help=(
            "Lyrics field name. Supports tolerant matching "
            "(case/space/BOM differences)."
        ),
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
        "--start-from",
        type=int,
        default=1,
        help="1-based non-empty JSONL record index to start processing from.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save checkpoint every N items.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume JSONL processing from existing output prefix. Enabled by default.",
    )
    parser.add_argument(
        "--check-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run API preflight checks before processing. Enabled by default.",
    )
    parser.add_argument(
        "--api-check-only",
        action="store_true",
        help="Only run API preflight checks and exit.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Allow overwriting existing output when --no-resume is used.",
    )
    return parser.parse_args()


def preflight_check_bailian_api(
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
) -> None:
    """Check API connectivity, authentication, and model availability.

    Raises:
        RuntimeError: If API check fails.
    """
    base = base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    models_url = f"{base}/models"
    try:
        response = requests.get(models_url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"API connectivity check failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"API auth/check failed ({response.status_code}) at {models_url}: {response.text[:300]}"
        )

    model_exists = False
    try:
        body = response.json()
        data = body.get("data", []) if isinstance(body, dict) else []
        for item in data:
            if isinstance(item, dict) and item.get("id") == model:
                model_exists = True
                break
    except Exception:
        # If models response is non-standard but HTTP is 200, we still allow processing.
        model_exists = True

    if not model_exists:
        print(
            f"Warning: model '{model}' not found in /models list. "
            "Processing may still work if provider aliases this model."
        )

    print(f"API preflight passed: base_url={base}, model={model}")


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
    def _normalized(name: str) -> str:
        return name.replace("\ufeff", "").strip().lower().replace("-", "_").replace(" ", "")

    normalized_to_real_key: Dict[str, str] = {}
    for key in record.keys():
        if isinstance(key, str):
            normalized_to_real_key[_normalized(key)] = key

    if preferred_field:
        if preferred_field in record and isinstance(record[preferred_field], (str, dict, list)):
            return preferred_field
        preferred_norm = _normalized(preferred_field)
        preferred_real = normalized_to_real_key.get(preferred_norm)
        if preferred_real and isinstance(record[preferred_real], (str, dict, list)):
            return preferred_real

    for field in LYRICS_CANDIDATE_FIELDS:
        if field in record and isinstance(record[field], (str, dict, list)):
            return field
        field_norm = _normalized(field)
        real_key = normalized_to_real_key.get(field_norm)
        if real_key and isinstance(record[real_key], (str, dict, list)):
            return real_key

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


def _has_processed_meta(record: Any) -> bool:
    """Return True when record has a valid processing metadata dict."""
    if not isinstance(record, dict):
        return False
    meta = record.get("lyrics_clean_meta")
    if not isinstance(meta, dict):
        return False
    return isinstance(meta.get("status"), str)


def _count_processed_prefix_jsonl(path: Path) -> int:
    """Count contiguous processed records from the beginning of a JSONL file."""
    if not path.exists():
        return 0

    processed = 0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                break
            if not _has_processed_meta(obj):
                break
            processed += 1
    return processed


def _trim_jsonl_to_prefix(path: Path, keep_records: int) -> None:
    """Keep only first N non-empty JSONL records in file."""
    if not path.exists():
        return

    kept_lines: List[str] = []
    kept = 0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if kept >= keep_records:
                break
            kept_lines.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            kept += 1

    with path.open("w", encoding="utf-8") as f:
        f.writelines(kept_lines)


def process_jsonl_stream(args: argparse.Namespace) -> None:
    """Process JSONL in streaming mode: read one line, process, then write one line."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.start_from < 1:
        raise ValueError("--start-from must be >= 1")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    total = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1

    manual_start_done = min(args.start_from - 1, total)
    target_end = total if args.max_items <= 0 else min(total, manual_start_done + args.max_items)
    print(
        f"Loaded {total} records, start_from={args.start_from}, "
        f"target_range={manual_start_done + 1}-{target_end}"
    )

    resume_done = 0
    write_mode = "w"
    if (not args.resume) and output_path.exists() and (not args.force_overwrite):
        raise RuntimeError(
            "Refusing to overwrite existing output with --no-resume. "
            "Use --resume, change --output, or add --force-overwrite explicitly."
        )

    if args.resume and output_path.exists():
        resume_done = _count_processed_prefix_jsonl(output_path)
        resume_done = min(resume_done, target_end)
        _trim_jsonl_to_prefix(output_path, resume_done)
        write_mode = "a"
        print(f"Resume enabled: already processed {resume_done}/{target_end}")

    effective_done = max(resume_done, manual_start_done)

    if effective_done >= target_end:
        print(f"Nothing to do. output={output_path}")
        return

    success = 0
    skipped = 0
    failed = 0
    skip_reason_counter: Counter[str] = Counter()
    missing_field_key_examples: List[List[str]] = []
    handled_in_scope = 0
    seen_non_empty = 0
    processed_total = effective_done

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(write_mode, encoding="utf-8") as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue

            seen_non_empty += 1
            if seen_non_empty <= resume_done:
                continue

            if seen_non_empty <= effective_done:
                # Backfill records skipped due to manual start when appending to an existing prefix.
                fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            if seen_non_empty > target_end:
                # Preserve trailing records when max-items limits processing range.
                fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            handled_in_scope += 1
            processed_total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                skip_reason_counter["invalid_json_line"] += 1
                fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            if not isinstance(record, dict):
                skipped += 1
                skip_reason_counter["non_dict_record"] += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            field = find_lyrics_field(record, args.lyrics_field)
            if not field:
                skipped += 1
                skip_reason_counter["lyrics_field_not_found"] += 1
                if len(missing_field_key_examples) < 3:
                    missing_field_key_examples.append([str(k) for k in list(record.keys())[:20]])
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
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
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
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
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if handled_in_scope % max(args.save_every, 1) == 0:
                fout.flush()
                print(f"Checkpoint saved at {processed_total}/{target_end}")

    print(
        "Done. "
        f"success={success}, skipped={skipped}, failed={failed}, output={output_path}"
    )
    if skip_reason_counter:
        print(f"Skip reasons: {dict(skip_reason_counter)}")
    if success == 0 and failed == 0 and skipped > 0:
        print(
            "Hint: all records were skipped because no lyrics field was detected. "
            "Try --lyrics-field lyric_text or update LYRICS_CANDIDATE_FIELDS."
        )
        if missing_field_key_examples:
            print("Sample keys from skipped records:")
            for i, key_list in enumerate(missing_field_key_examples, start=1):
                print(f"  [{i}] {key_list}")


def process_records(args: argparse.Namespace) -> None:
    """Process all records and write cleaned dataset."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".jsonl":
        process_jsonl_stream(args)
        return

    records, root_obj, root_list_key = load_dataset(input_path)
    total = len(records)
    limit = args.max_items if args.max_items > 0 else total
    limit = min(limit, total)

    print(f"Loaded {total} records, processing {limit} records")

    success = 0
    skipped = 0
    failed = 0
    skip_reason_counter: Counter[str] = Counter()
    missing_field_key_examples: List[List[str]] = []

    for idx in range(limit):
        record = records[idx]
        if not isinstance(record, dict):
            skipped += 1
            skip_reason_counter["non_dict_record"] += 1
            continue

        field = find_lyrics_field(record, args.lyrics_field)
        if not field:
            skipped += 1
            skip_reason_counter["lyrics_field_not_found"] += 1
            if len(missing_field_key_examples) < 3:
                missing_field_key_examples.append(
                    [str(k) for k in list(record.keys())[:20]]
                )
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
    if skip_reason_counter:
        print(f"Skip reasons: {dict(skip_reason_counter)}")
    if success == 0 and failed == 0 and skipped > 0:
        print(
            "Hint: all records were skipped because no lyrics field was detected. "
            "Try --lyrics-field lyric_text or update LYRICS_CANDIDATE_FIELDS."
        )
        if missing_field_key_examples:
            print("Sample keys from skipped records:")
            for i, key_list in enumerate(missing_field_key_examples, start=1):
                print(f"  [{i}] {key_list}")


def main() -> None:
    """CLI entry."""
    args = parse_args()

    if args.check_api or args.api_check_only:
        preflight_check_bailian_api(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
        )

    if args.api_check_only:
        return

    process_records(args)


if __name__ == "__main__":
    main()
