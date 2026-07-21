"""
Audio file discovery and metadata loading for preprocessing.

Reads both per-sample metadata and dataset-level metadata (``tag_position``,
``genre_ratio``, default ``custom_tag``) from ACE-Step's JSON format.

Extracted from ``preprocess.py`` to keep that module under the LOC limit.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Supported audio extensions (same as upstream)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def discover_audio_files(
    audio_dir: Optional[str],
    dataset_json: Optional[str],
) -> List[Path]:
    """Discover audio files from a dataset JSON or by scanning a directory.

    Resolution order:

    1. If *dataset_json* is provided, extract ``audio_path`` (or fall back
       to ``filename``) from each entry.  Missing files are skipped with a
       warning.
    2. Otherwise, recursively scan *audio_dir* for supported audio
       extensions (``AUDIO_EXTENSIONS``).
    """
    # -- JSON-driven discovery ----------------------------------------------
    if dataset_json and Path(dataset_json).is_file():
        try:
            raw = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
            samples = raw if isinstance(raw, list) else raw.get("samples", [])
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[Side-Step] Failed to read dataset JSON: %s", exc)
            samples = []

        audio_files: List[Path] = []
        json_dir = Path(dataset_json).parent  # resolve relative paths vs JSON
        for entry in samples:
            ap = entry.get("audio_path") or entry.get("filename", "")
            if not ap:
                continue
            p = Path(ap)
            if not p.is_absolute():
                p = json_dir / p
            if p.is_file():
                audio_files.append(p)
            else:
                logger.warning("[Side-Step] Audio file from JSON not found: %s", p)

        if audio_files:
            logger.info(
                "[Side-Step] Resolved %d audio files from dataset JSON", len(audio_files),
            )
            return sorted(audio_files)
        else:
            logger.warning(
                "[Side-Step] Dataset JSON contained no resolvable audio paths; "
                "falling back to directory scan"
            )

    # -- Recursive directory scan -------------------------------------------
    if not audio_dir:
        return []

    source_path = Path(audio_dir)
    if not source_path.is_dir():
        logger.warning("[Side-Step] Audio directory does not exist: %s", audio_dir)
        return []

    audio_files = sorted(
        f for f in source_path.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if audio_files:
        logger.info(
            "[Side-Step] Found %d audio files (recursive scan of %s)",
            len(audio_files), audio_dir,
        )
    return audio_files


def load_sample_metadata(
    dataset_json: Optional[str],
    audio_files: List[Path],
) -> Dict[str, Dict[str, Any]]:
    """Build a filename -> metadata mapping.

    If *dataset_json* is provided, load it and index by filename.
    Falls back to basename of ``audio_path`` when ``filename`` is missing.
    Otherwise return defaults for every audio file.
    """
    meta: Dict[str, Dict[str, Any]] = {}

    if dataset_json and Path(dataset_json).is_file():
        try:
            raw = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
            samples = raw if isinstance(raw, list) else raw.get("samples", [])
            for s in samples:
                # Primary key: explicit filename field
                fname = s.get("filename", "")
                if fname:
                    meta[fname] = s
                    # Also index by basename if filename contains a path
                    basename = Path(fname).name
                    if basename != fname and basename not in meta:
                        meta[basename] = s
                elif s.get("audio_path"):
                    # Fallback: derive key from audio_path basename
                    basename = Path(s["audio_path"]).name
                    if basename and basename not in meta:
                        meta[basename] = s
            logger.info("[Side-Step] Loaded metadata for %d samples from %s", len(meta), dataset_json)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[Side-Step] Failed to load dataset JSON: %s", exc)

    # Fill defaults for any audio file without metadata.
    # Also scan for sidecar .lyrics.txt / .caption.txt and .json files
    # (same behaviour as the V1 ScanMixin pipeline).
    for af in audio_files:
        if af.name not in meta:
            meta[af.name] = _build_sample_meta_from_files(af)
        else:
            # Sample exists in JSON – fill in any missing fields from sidecar files.
            _patch_meta_from_files(af, meta[af.name])

    return meta


def _read_sidecar_text(path: str) -> Optional[str]:
    """Read a sidecar text file, returning stripped content or None."""
    if not os.path.isfile(path):
        return None
    try:
        content = Path(path).read_text(encoding="utf-8").strip()
        return content if content else None
    except OSError:
        return None


def _build_sample_meta_from_files(af: Path) -> Dict[str, Any]:
    """Build sample metadata from sidecar files (no JSON entry)."""
    base = os.path.splitext(str(af))[0]
    lyrics = "[Instrumental]"
    caption = af.stem.replace("_", " ").replace("-", " ")
    is_instrumental = True

    # Lyrics: .lyrics.txt (preferred), then .txt (legacy)
    for suffix in (".lyrics.txt", ".txt"):
        content = _read_sidecar_text(base + suffix)
        if content is not None:
            lyrics = content
            is_instrumental = False
            break

    # Caption: .caption.txt
    caption_content = _read_sidecar_text(base + ".caption.txt")
    if caption_content is not None:
        caption = caption_content

    # JSON sidecar: bpm / keyscale / timesignature / language
    json_meta = {}
    json_path = base + ".json"
    if os.path.isfile(json_path):
        try:
            json_meta = json.loads(Path(json_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if json_meta.get("language") and json_meta.get("language") != "instrumental":
        is_instrumental = False

    return {
        "filename": af.name,
        "caption": caption,
        "lyrics": lyrics,
        "genre": json_meta.get("genre", ""),
        "bpm": json_meta.get("bpm"),
        "keyscale": json_meta.get("keyscale", ""),
        "timesignature": json_meta.get("timesignature", ""),
        "duration": json_meta.get("duration", 0),
        "is_instrumental": is_instrumental,
    }


def _patch_meta_from_files(af: Path, meta: Dict[str, Any]) -> None:
    """Fill missing lyrics / caption / bpm from sidecar files when JSON entry exists."""
    base = os.path.splitext(str(af))[0]

    # Only fill lyrics if the JSON didn't provide them
    if not meta.get("lyrics") or meta["lyrics"] == "[Instrumental]":
        for suffix in (".lyrics.txt", ".txt"):
            content = _read_sidecar_text(base + suffix)
            if content is not None:
                meta["lyrics"] = content
                meta["is_instrumental"] = False
                break

    # Only fill caption if JSON didn't provide it
    if not meta.get("caption"):
        caption_content = _read_sidecar_text(base + ".caption.txt")
        if caption_content is not None:
            meta["caption"] = caption_content


def load_dataset_metadata(dataset_json: Optional[str]) -> Dict[str, Any]:
    """Load the top-level ``metadata`` block from an ACE-Step dataset JSON.

    Returns a dict with the dataset-level settings that affect prompt
    construction:

    - ``tag_position`` (str): ``"prepend"``, ``"append"``, or ``"replace"``.
    - ``genre_ratio`` (int): 0-100, percentage of samples using genre.
    - ``custom_tag`` (str): default trigger word applied to all samples.

    Returns safe defaults when the JSON has no metadata block or is absent.
    """
    defaults: Dict[str, Any] = {
        "tag_position": "prepend",
        "genre_ratio": 0,
        "custom_tag": "",
    }
    if not dataset_json or not Path(dataset_json).is_file():
        return defaults

    try:
        raw = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return defaults

    if not isinstance(raw, dict) or "metadata" not in raw:
        return defaults

    meta = raw["metadata"]
    return {
        "tag_position": meta.get("tag_position", "prepend"),
        "genre_ratio": meta.get("genre_ratio", 0),
        "custom_tag": meta.get("custom_tag", ""),
    }


def select_genre_indices(num_samples: int, genre_ratio: int) -> Set[int]:
    """Select sample indices that should use genre instead of caption.

    Mirrors upstream ``select_genre_indices()`` from ACE-Step's
    ``preprocess_utils.py``.  Uses a fixed seed so the selection is
    deterministic and reproducible across runs.

    Args:
        num_samples: Total number of samples.
        genre_ratio: 0-100, percentage of samples that use genre.

    Returns:
        Set of sample indices that should use genre.
    """
    if genre_ratio <= 0 or num_samples <= 0:
        return set()
    num_genre = int(num_samples * genre_ratio / 100)
    rng = random.Random(42)
    indices = list(range(num_samples))
    rng.shuffle(indices)
    return set(indices[:num_genre])
