#!/usr/bin/env python3
"""
Scan an audio directory, detect actual codec via ffprobe,
convert non-MP3 files to MP3, and replace the originals.
"""

import json
import subprocess
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".mp3", ".m4a", ".ape", ".aiff"}


def get_codec(filepath: Path) -> str | None:
    """Use ffprobe to detect the actual audio codec of a file."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "json",
            str(filepath),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        return data["streams"][0]["codec_name"]
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


def convert_to_mp3(src: Path) -> Path | None:
    """Convert audio file to MP3 via ffmpeg, return output path."""
    dst = src.with_suffix(".mp3")

    print(f"  [{src.name}] -> [{dst.name}] converting ...")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-acodec", "libmp3lame", "-q:a", "2",
            str(dst),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        last_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        print(f"    [FAIL] {last_line}")
        return None

    # Delete original after successful conversion
    try:
        src.unlink()
    except OSError as e:
        print(f"    [WARN] could not delete original: {e}")

    print(f"    [OK] converted and replaced")
    return dst


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <audio_directory>")
        sys.exit(1)

    audio_dir = Path(sys.argv[1])
    if not audio_dir.is_dir():
        print(f"Error: {audio_dir} is not a directory")
        sys.exit(1)

    # Collect all audio files
    audio_files = sorted(
        f for f in audio_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print("No audio files found.")
        return

    # Detect codec for each file
    to_convert: list[Path] = []
    skip_same_stem_mp3: set[str] = {f.stem for f in audio_files if f.suffix.lower() == ".mp3"}

    for f in audio_files:
        if f.suffix.lower() == ".mp3":
            continue  # already MP3 extension, skip

        # If an .mp3 with the same stem already exists, skip
        if f.stem in skip_same_stem_mp3:
            print(f"  [{f.name}] SKIP — {f.stem}.mp3 already exists")
            continue

        codec = get_codec(f)
        if codec is None:
            print(f"  [{f.name}] SKIP — ffprobe failed")
            continue

        if codec == "mp3":
            print(f"  [{f.name}] already MP3 codec ({codec}), SKIP")
        else:
            print(f"  [{f.name}] codec={codec} -> will convert")
            to_convert.append(f)

    if not to_convert:
        print("\nAll files are already MP3-encoded. Nothing to do.")
        return

    print(f"\nConverting {len(to_convert)} file(s):\n")
    converted = 0
    for f in to_convert:
        if convert_to_mp3(f):
            converted += 1

    print(f"\nDone: {converted}/{len(to_convert)} converted.")


if __name__ == "__main__":
    main()
