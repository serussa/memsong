"""Repair music caption tags with Bailian LLM API.

This script fixes tag quality for genre/vocal/instruments/mood while preserving
tempo and key tags exactly as they appear in the original caption.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import mimetypes
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


BAILIAN_API_KEY = "sk-d06075dbd6914fa797f9bd3787a96e34"
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BAILIAN_MODEL = "qwen3.5-omni-plus-2026-03-15"

DEFAULT_INPUT_PATH = "/root/autodl-tmp/musicdata/dataset_tags_standardized.jsonl"
DEFAULT_OUTPUT_PATH = "/root/autodl-tmp/musicdata/dataset_tags_repaired.jsonl"
DEFAULT_AUDIO_ROOT = "/root/autodl-tmp/musicdata/audios"

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".audio")

TEMPO_PATTERNS = [
	r"(?i)^\d{2,3}\s*bpm$",
	r"(?i)^(very\s+)?(slow|mid|medium|fast)\s+tempo$",
]

KEY_PATTERNS = [
	r"(?i)^[A-G](?:#|b)?\s*(major|minor)$",
	r"(?i)^(major|minor)\s+key$",
]


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments."""
	parser = argparse.ArgumentParser(
		description="Repair caption tags while preserving tempo/key"
	)
	parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input JSONL path")
	parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSONL path")
	parser.add_argument(
		"--audio-root",
		default=DEFAULT_AUDIO_ROOT,
		help="Base directory for resolving relative audio paths",
	)
	parser.add_argument(
		"--audio-field",
		default="",
		help=(
			"Audio path field name. Empty means auto-detect from common names or "
			"single-key record key when it looks like an audio filename."
		),
	)
	parser.add_argument(
		"--caption-field",
		default="",
		help=(
			"Caption field name. Empty means auto-detect from first string value "
			"or the only key in record."
		),
	)
	parser.add_argument(
		"--write-field",
		default="",
		help="Field to write repaired caption. Empty means overwrite detected field.",
	)
	parser.add_argument("--model", default=BAILIAN_MODEL, help="Bailian model name")
	parser.add_argument(
		"--api-key",
		default="sk-d06075dbd6914fa797f9bd3787a96e34",
		help="Bailian API key",
	)
	parser.add_argument("--base-url", default=BAILIAN_BASE_URL, help="Bailian base URL")
	parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
	parser.add_argument(
		"--segment-count",
		type=int,
		default=3,
		help="How many short audio segments to sample for inference.",
	)
	parser.add_argument(
		"--segment-duration",
		type=float,
		default=12.0,
		help="Duration (seconds) per sampled segment.",
	)
	parser.add_argument(
		"--request-retries",
		type=int,
		default=3,
		help="How many retry rounds for API requests.",
	)
	parser.add_argument(
		"--retry-delay",
		type=float,
		default=1.5,
		help="Base delay seconds between retries (linear backoff).",
	)
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
		default=1,
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
	"""Check API connectivity, authentication, and model availability."""
	if not api_key.strip():
		raise RuntimeError("Missing API key. Provide --api-key or set BAILIAN_API_KEY.")

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
			f"API auth/check failed ({response.status_code}) at {models_url}: "
			f"{response.text[:300]}"
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
		model_exists = True

	if not model_exists:
		print(
			f"Warning: model '{model}' not found in /models list. "
			"Processing may still work if provider aliases this model."
		)

	print(f"API preflight passed: base_url={base}, model={model}")


def extract_json_object(text: str) -> Dict[str, Any]:
	"""Extract the first valid JSON object from model output."""
	payload = text.strip()
	try:
		return json.loads(payload)
	except json.JSONDecodeError:
		pass

	start = payload.find("{")
	if start == -1:
		raise ValueError("Model output does not contain JSON object")

	depth = 0
	in_string = False
	escape = False
	end = -1

	for i, ch in enumerate(payload[start:], start=start):
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

	return json.loads(payload[start:end])


def _normalized(name: str) -> str:
	"""Normalize field name for tolerant matching."""
	return name.replace("\ufeff", "").strip().lower().replace("-", "_").replace(" ", "")


def find_caption_field(record: Dict[str, Any], preferred_field: str = "") -> Optional[str]:
	"""Find caption field from preferred name or first string-like value."""
	if preferred_field:
		if preferred_field in record and isinstance(record[preferred_field], (str, list, dict)):
			return preferred_field
		norm_map = {
			_normalized(k): k for k in record.keys() if isinstance(k, str)
		}
		real_key = norm_map.get(_normalized(preferred_field))
		if real_key and isinstance(record[real_key], (str, list, dict)):
			return real_key

	if len(record) == 1:
		only_key = next(iter(record.keys()))
		if isinstance(record.get(only_key), (str, list, dict)):
			return str(only_key)

	for key, value in record.items():
		if isinstance(value, (str, list, dict)):
			return str(key)
	return None


def to_caption_text(value: Any) -> str:
	"""Convert caption payload to text."""
	if isinstance(value, str):
		return value
	if isinstance(value, list):
		return ", ".join(str(v).strip() for v in value if str(v).strip())
	return json.dumps(value, ensure_ascii=False)


def _looks_like_audio_filename(name: str) -> bool:
	"""Return True when name looks like an audio filename/path."""
	lower_name = name.strip().lower()
	return lower_name.endswith(AUDIO_EXTENSIONS)


def _resolve_audio_path(audio_ref: str, audio_root: str) -> Path:
	"""Resolve audio reference to an absolute path."""
	audio_path = Path(audio_ref)
	if audio_path.is_absolute():
		return audio_path
	return Path(audio_root) / audio_path


def find_audio_ref(record: Dict[str, Any], preferred_field: str = "") -> Optional[str]:
	"""Find audio reference from preferred field, common fields, or single-key record key."""
	if preferred_field and preferred_field in record and isinstance(record[preferred_field], str):
		if record[preferred_field].strip():
			return record[preferred_field].strip()

	common_fields = [
		"audio_path",
		"path",
		"audio",
		"wav",
		"file",
		"filename",
		"audio_file",
	]
	for field in common_fields:
		if field in record and isinstance(record[field], str) and record[field].strip():
			return record[field].strip()

	if len(record) == 1:
		only_key = str(next(iter(record.keys())))
		if _looks_like_audio_filename(only_key):
			return only_key

	for key in record.keys():
		if isinstance(key, str) and _looks_like_audio_filename(key):
			return key
	return None


def _audio_format_from_path(path: Path) -> str:
	"""Map file suffix to API audio format value."""
	suffix = path.suffix.lower().lstrip(".")
	if suffix == "m4a":
		return "mp4"
	if suffix:
		return suffix
	return "mp3"


def _build_data_url(path: Path) -> str:
	"""Build data URL from local audio file."""
	mime_type = mimetypes.guess_type(str(path))[0] or "audio/mpeg"
	with path.open("rb") as f:
		raw = f.read()
	encoded = base64.b64encode(raw).decode("ascii")
	return f"data:{mime_type};base64,{encoded}"


def _build_video_data_url_from_audio(path: Path) -> str:
	"""Build mp4 video data URL from an audio file using ffmpeg.

	DashScope chat/completions in this workflow supports text/image/video types,
	so audio is wrapped into a short black-frame mp4 with the original audio track.
	"""
	ffmpeg = shutil.which("ffmpeg")
	if not ffmpeg:
		raise RuntimeError("ffmpeg is required to convert audio clips into video_url payload")

	tmp_fd, tmp_name = tempfile.mkstemp(prefix="caption_media_", suffix=".mp4")
	Path(tmp_name).unlink(missing_ok=True)
	try:
		cmd = [
			ffmpeg,
			"-y",
			"-v",
			"error",
			"-f",
			"lavfi",
			"-i",
			"color=size=16x16:rate=1:color=black",
			"-i",
			str(path),
			"-shortest",
			"-c:v",
			"libx264",
			"-preset",
			"ultrafast",
			"-pix_fmt",
			"yuv420p",
			"-c:a",
			"aac",
			"-b:a",
			"64k",
			tmp_name,
		]
		subprocess.run(cmd, check=True, capture_output=True, text=True)
		return _build_data_url(Path(tmp_name))
	except subprocess.SubprocessError as exc:
		raise RuntimeError(f"Failed to convert audio clip to mp4: {exc}") from exc
	finally:
		Path(tmp_name).unlink(missing_ok=True)


def _probe_audio_duration_seconds(audio_path: Path) -> Optional[float]:
	"""Probe audio duration with ffprobe; return None when unavailable."""
	ffprobe = shutil.which("ffprobe")
	if not ffprobe:
		return None

	cmd = [
		ffprobe,
		"-v",
		"error",
		"-show_entries",
		"format=duration",
		"-of",
		"default=noprint_wrappers=1:nokey=1",
		str(audio_path),
	]
	try:
		result = subprocess.run(cmd, check=True, capture_output=True, text=True)
		duration_text = result.stdout.strip()
		if not duration_text:
			return None
		value = float(duration_text)
		return value if value > 0 else None
	except (subprocess.SubprocessError, ValueError):
		return None


def _extract_audio_segment(
	input_audio_path: Path,
	output_audio_path: Path,
	start_sec: float,
	duration_sec: float,
) -> bool:
	"""Extract and downsample one short segment with ffmpeg."""
	ffmpeg = shutil.which("ffmpeg")
	if not ffmpeg:
		return False

	cmd = [
		ffmpeg,
		"-y",
		"-v",
		"error",
		"-ss",
		f"{max(start_sec, 0.0):.3f}",
		"-t",
		f"{max(duration_sec, 1.0):.3f}",
		"-i",
		str(input_audio_path),
		"-ac",
		"1",
		"-ar",
		"16000",
		"-vn",
		str(output_audio_path),
	]
	try:
		subprocess.run(cmd, check=True, capture_output=True, text=True)
		return output_audio_path.exists() and output_audio_path.stat().st_size > 0
	except subprocess.SubprocessError:
		return False


def prepare_audio_clips(
	audio_path: Path,
	segment_count: int,
	segment_duration: float,
) -> Tuple[List[Path], Optional[Path]]:
	"""Prepare sampled clips for multimodal inference.

	Returns clip paths and optional temp directory that should be cleaned up.
	"""
	if segment_count <= 1 or segment_duration <= 0:
		return [audio_path], None

	ffmpeg_ok = bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))
	if not ffmpeg_ok:
		raise RuntimeError("Audio slicing requires ffmpeg and ffprobe in PATH")

	duration = _probe_audio_duration_seconds(audio_path)
	if duration is None:
		raise RuntimeError(f"Failed to probe audio duration: {audio_path}")

	if duration <= segment_duration * 1.2:
		return [audio_path], None

	centers = [0.25, 0.50, 0.72, 0.85]
	use_count = max(1, min(segment_count, len(centers)))
	temp_dir = Path(tempfile.mkdtemp(prefix="caption_segments_"))
	clip_paths: List[Path] = []

	for idx in range(use_count):
		center_ratio = centers[idx]
		center = duration * center_ratio
		start = max(0.0, min(center - segment_duration / 2.0, duration - segment_duration))
		out_path = temp_dir / f"seg_{idx + 1}.wav"
		if _extract_audio_segment(audio_path, out_path, start, segment_duration):
			clip_paths.append(out_path)

	if not clip_paths:
		shutil.rmtree(temp_dir, ignore_errors=True)
		raise RuntimeError(f"Failed to extract any audio segment from: {audio_path}")

	if len(clip_paths) < use_count:
		shutil.rmtree(temp_dir, ignore_errors=True)
		raise RuntimeError(
			f"Only extracted {len(clip_paths)}/{use_count} audio segments from: {audio_path}"
		)

	return clip_paths, temp_dir


def split_tags(caption_text: str) -> List[str]:
	"""Split comma-separated tags and remove empty parts."""
	return [part.strip() for part in caption_text.split(",") if part.strip()]


def is_tempo_tag(tag: str) -> bool:
	"""Return True when tag represents tempo information."""
	return any(re.match(pattern, tag.strip()) for pattern in TEMPO_PATTERNS)


def is_key_tag(tag: str) -> bool:
	"""Return True when tag represents key information."""
	return any(re.match(pattern, tag.strip()) for pattern in KEY_PATTERNS)


def split_protected_tags(tags: List[str]) -> Tuple[List[str], List[str]]:
	"""Split tags into protected (tempo/key) and editable groups."""
	protected: List[str] = []
	editable: List[str] = []
	for tag in tags:
		if is_tempo_tag(tag) or is_key_tag(tag):
			protected.append(tag)
		else:
			editable.append(tag)
	return protected, editable


def dedupe_preserve_order(tags: List[str]) -> List[str]:
	"""Remove duplicate tags while preserving first occurrence order."""
	seen = set()
	result: List[str] = []
	for tag in tags:
		norm = tag.strip().lower()
		if not norm or norm in seen:
			continue
		seen.add(norm)
		result.append(tag.strip())
	return result


def call_bailian_repair(
	editable_tags: List[str],
	full_tags: List[str],
	audio_paths: List[Path],
	api_key: str,
	model: str,
	base_url: str,
	timeout: int,
	request_retries: int,
	retry_delay: float,
) -> Dict[str, Any]:
	"""Call Bailian multimodal API to repair tags based on audio content."""
	system_prompt = (
		"你是音乐标签修复助手。"
		"你会收到一段音频和一组粗糙标签。"
		"请根据音频实际可听内容进行判断，不要只机械复述粗标签。"
		"请只修复这些类别：genre、vocal type、instruments、mood/energy。"
		"Guidelines for creating prompt tags: "
		"Include genre (e.g., 'rap', 'pop', 'rock', 'electronic'). "
		"Include vocal type (e.g., 'male vocal', 'female vocal', 'spoken word'). "
		"Include instruments actually heard (e.g., 'guitar', 'piano', 'synthesizer', 'drums'). "
		"Include mood/energy (e.g., 'energetic', 'calm', 'aggressive', 'melancholic'). "
		"不要生成 tempo/key，tempo/key 会由外部逻辑保留原值。"
		"保持简洁英文标签，逗号分隔，不要输出解释文本。"
		"若原标签已经合理，尽量少改动。"
		"只输出 JSON，不要 markdown 代码块。"
	)
	user_prompt = (
		"请基于音频内容修复以下标签（只修复 genre/vocal/instruments/mood）。\n"
		"请严格遵循：\n"
		"1) Include genre (e.g., 'rap', 'pop', 'rock', 'electronic')\n"
		"2) Include vocal type (e.g., 'male vocal', 'female vocal', 'spoken word')\n"
		"3) Include instruments actually heard (e.g., 'guitar', 'piano', 'synthesizer', 'drums')\n"
		"4) Include mood/energy (e.g., 'energetic', 'calm', 'aggressive', 'melancholic')\n\n"
		"完整原始标签（仅供参考）：\n"
		f"{', '.join(full_tags)}\n\n"
		"可编辑标签（tempo/key 已移除）：\n"
		f"{', '.join(editable_tags)}\n\n"
		"请输出 JSON，严格格式：\n"
		"{\n"
		"  \"fixed_tags\": \"<仅包含 genre/vocal/instruments/mood 的英文标签，逗号分隔；优先体现以上四类信息>\",\n"
		"  \"change_notes\": \"<简短说明>\"\n"
		"}"
	)

	url = f"{base_url.rstrip('/')}/chat/completions"
	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json",
	}
	if not audio_paths:
		raise RuntimeError("No audio input for multimodal inference")

	media_items_video_url: List[Dict[str, Any]] = []
	for path in audio_paths:
		data_url = _build_video_data_url_from_audio(path)
		media_items_video_url.append({"type": "video_url", "video_url": {"url": data_url}})

	payload = {
		"model": model,
		"temperature": 0.1,
		"messages": [
			{"role": "system", "content": system_prompt},
			{
				"role": "user",
				"content": [{"type": "text", "text": user_prompt}] + media_items_video_url,
			},
		],
		"response_format": {"type": "json_object"},
	}

	last_error: Optional[str] = None
	total_rounds = max(1, request_retries)
	for round_idx in range(total_rounds):
		try:
			response = requests.post(url, headers=headers, json=payload, timeout=timeout)
		except requests.RequestException as exc:
			last_error = str(exc)
			response = None

		if response is not None and response.status_code == 200:
			body = response.json()
			content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
			if content:
				result = extract_json_object(content)
				if "fixed_tags" in result:
					return result
				last_error = "Model JSON missing 'fixed_tags'"
			else:
				last_error = "Empty content from Bailian API"
		elif response is not None:
			last_error = f"{response.status_code}: {response.text[:500]}"

		if round_idx < total_rounds - 1:
			time.sleep(max(0.0, retry_delay) * (round_idx + 1))

	raise RuntimeError(f"Bailian multimodal API error: {last_error or 'unknown'}")


def _has_processed_meta(record: Any) -> bool:
	"""Return True when record has finished processing metadata.

	Records with ``status=error`` are considered unfinished so resume can retry them.
	"""
	if not isinstance(record, dict):
		return False
	meta = record.get("caption_tag_fix_meta")
	if not isinstance(meta, dict):
		return False
	status = meta.get("status")
	if not isinstance(status, str):
		return False
	return status.strip().lower() != "error"


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
	"""Process JSONL in streaming mode and repair caption tags per record."""
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
	if manual_start_done >= target_end:
		print(f"Nothing to do. output={output_path}")
		return

	existing_output_lines: List[str] = []
	if (not args.resume) and output_path.exists() and (not args.force_overwrite):
		raise RuntimeError(
			"Refusing to overwrite existing output with --no-resume. "
			"Use --resume, change --output, or add --force-overwrite explicitly."
		)

	if args.resume and output_path.exists():
		with output_path.open("r", encoding="utf-8") as f:
			for raw_line in f:
				line = raw_line.strip()
				if not line:
					continue
				existing_output_lines.append(
					raw_line if raw_line.endswith("\n") else raw_line + "\n"
				)
		print(
			"Resume enabled: "
			f"loaded {len(existing_output_lines)} existing non-empty records from output"
		)

	success = 0
	skipped = 0
	failed = 0
	reused = 0
	skip_reason_counter: Counter[str] = Counter()
	handled_in_scope = 0
	seen_non_empty = 0

	output_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_output_path = output_path.with_name(output_path.name + ".tmp")
	try:
		with input_path.open("r", encoding="utf-8") as fin, tmp_output_path.open("w", encoding="utf-8") as fout:
			for raw_line in fin:
				line = raw_line.strip()
				if not line:
					continue

				seen_non_empty += 1
				existing_raw = (
					existing_output_lines[seen_non_empty - 1]
					if seen_non_empty <= len(existing_output_lines)
					else None
				)

				if seen_non_empty <= manual_start_done or seen_non_empty > target_end:
					if existing_raw is not None:
						fout.write(existing_raw)
					else:
						fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
					continue

				handled_in_scope += 1

				if args.resume and existing_raw is not None:
					try:
						existing_obj = json.loads(existing_raw.strip())
					except json.JSONDecodeError:
						existing_obj = None
					if _has_processed_meta(existing_obj):
						reused += 1
						fout.write(existing_raw)
						if handled_in_scope % max(args.save_every, 1) == 0:
							fout.flush()
							print(f"Checkpoint saved at {seen_non_empty}/{target_end}")
						continue

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

				field = find_caption_field(record, args.caption_field)
				if not field:
					skipped += 1
					skip_reason_counter["caption_field_not_found"] += 1
					fout.write(json.dumps(record, ensure_ascii=False) + "\n")
					continue

				write_field = args.write_field or field
				audio_ref = find_audio_ref(record, args.audio_field)
				if not audio_ref:
					skipped += 1
					skip_reason_counter["audio_path_not_found"] += 1
					record["caption_tag_fix_meta"] = {
						"status": "skipped",
						"reason": "audio_path_not_found",
						"source_field": field,
						"target_field": write_field,
					}
					fout.write(json.dumps(record, ensure_ascii=False) + "\n")
					continue

				resolved_audio_path = _resolve_audio_path(audio_ref, args.audio_root)
				if not resolved_audio_path.exists() or not resolved_audio_path.is_file():
					skipped += 1
					skip_reason_counter["audio_file_missing"] += 1
					record["caption_tag_fix_meta"] = {
						"status": "skipped",
						"reason": "audio_file_missing",
						"audio_ref": audio_ref,
						"audio_path": str(resolved_audio_path),
						"source_field": field,
						"target_field": write_field,
					}
					fout.write(json.dumps(record, ensure_ascii=False) + "\n")
					continue

				original_caption = to_caption_text(record[field]).strip()
				original_tags = split_tags(original_caption)

				if not original_tags:
					skipped += 1
					skip_reason_counter["empty_caption"] += 1
					record["caption_tag_fix_meta"] = {
						"status": "skipped",
						"reason": "empty_caption",
						"source_field": field,
						"target_field": write_field,
					}
					fout.write(json.dumps(record, ensure_ascii=False) + "\n")
					continue

				protected_tags, editable_tags = split_protected_tags(original_tags)

				if not editable_tags:
					record[write_field] = ", ".join(original_tags)
					record["caption_tag_fix_meta"] = {
						"status": "no_editable_tags",
						"source_field": field,
						"target_field": write_field,
						"protected_tags": protected_tags,
					}
					success += 1
					fout.write(json.dumps(record, ensure_ascii=False) + "\n")
					continue

				audio_clip_paths: List[Path] = [resolved_audio_path]
				temp_clip_dir: Optional[Path] = None

				try:
					audio_clip_paths, temp_clip_dir = prepare_audio_clips(
						audio_path=resolved_audio_path,
						segment_count=args.segment_count,
						segment_duration=args.segment_duration,
					)

					llm_result = call_bailian_repair(
						editable_tags=editable_tags,
						full_tags=original_tags,
						audio_paths=audio_clip_paths,
						api_key=args.api_key,
						model=args.model,
						base_url=args.base_url,
						timeout=args.timeout,
						request_retries=args.request_retries,
						retry_delay=args.retry_delay,
					)
					fixed_text = str(llm_result.get("fixed_tags", "")).strip()
					fixed_tags = split_tags(fixed_text)
					fixed_editable = [
						tag for tag in fixed_tags if not is_tempo_tag(tag) and not is_key_tag(tag)
					]
					fixed_editable = dedupe_preserve_order(fixed_editable)
					if not fixed_editable:
						fixed_editable = dedupe_preserve_order(editable_tags)

					merged_tags = dedupe_preserve_order(fixed_editable + protected_tags)
					record[write_field] = ", ".join(merged_tags)
					record["caption_tag_fix_meta"] = {
						"status": "ok",
						"change_notes": str(llm_result.get("change_notes", "")),
						"audio_ref": audio_ref,
						"audio_path": str(resolved_audio_path),
						"audio_clip_count": len(audio_clip_paths),
						"source_field": field,
						"target_field": write_field,
						"protected_tags": protected_tags,
						"editable_before": editable_tags,
						"editable_after": fixed_editable,
					}
					success += 1
				except Exception as exc:
					merged_tags = dedupe_preserve_order(editable_tags + protected_tags)
					record[write_field] = ", ".join(merged_tags)
					record["caption_tag_fix_meta"] = {
						"status": "error",
						"error": str(exc),
						"audio_ref": audio_ref,
						"audio_path": str(resolved_audio_path),
						"audio_clip_count": len(audio_clip_paths),
						"source_field": field,
						"target_field": write_field,
						"protected_tags": protected_tags,
						"change_notes": "LLM failed; fallback to original editable tags + protected tags.",
					}
					failed += 1
				finally:
					if temp_clip_dir is not None:
						shutil.rmtree(temp_clip_dir, ignore_errors=True)

				fout.write(json.dumps(record, ensure_ascii=False) + "\n")

				if handled_in_scope % max(args.save_every, 1) == 0:
					fout.flush()
					print(f"Checkpoint saved at {seen_non_empty}/{target_end}")

		tmp_output_path.replace(output_path)
	except Exception:
		tmp_output_path.unlink(missing_ok=True)
		raise

	print(
		"Done. "
		f"success={success}, skipped={skipped}, failed={failed}, reused={reused}, "
		f"output={output_path}"
	)
	if skip_reason_counter:
		print(f"Skip reasons: {dict(skip_reason_counter)}")


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

	process_jsonl_stream(args)


if __name__ == "__main__":
	main()
