#!/usr/bin/env python3
"""
Offline script to add section_ids to preprocessed .pt files.

Reads each .pt file's metadata["lyrics"], parses section structure,
and adds a "section_ids" tensor aligned to encoder_hidden_states.

Usage:
    python -m acestep.tgca.preprocess_section_ids \\
        --tensor-dir /root/autodl-tmp/musicdata/train_tensors \\
        --num-workers 8
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from acestep.tgca.lyrics_parser import LyricsStructureParser, SECTION_VOCAB

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("preprocess_section_ids")


def process_file(path: str, dry_run: bool = False) -> Optional[str]:
    """Add section_ids to a single .pt file. Returns path on success or error msg."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        metadata = data.get("metadata", {})
        lyrics_text = metadata.get("lyrics", "") if isinstance(metadata, dict) else ""
        L = data["encoder_hidden_states"].shape[0]

        parser = LyricsStructureParser()
        output = parser.parse(lyrics_text, num_chunks=L)
        section_ids = output.section_type_ids.to(torch.long)

        # Ensure correct length
        if section_ids.shape[0] != L:
            section_ids = torch.full((L,), 0, dtype=torch.long)  # all UNKNOWN

        if dry_run:
            num_unknown = (section_ids == 0).sum().item()
            if output.num_sections > 0:
                return f"{os.path.basename(path)}: {output.num_sections} sections, {L} tokens, {num_unknown}/{L} UNKNOWN"
            return None  # no sections, skip

        data["section_ids"] = section_ids
        torch.save(data, path)
        return f"{os.path.basename(path)}: added section_ids [{L}]"

    except Exception as e:
        return f"ERROR {os.path.basename(path)}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dir", default="/root/autodl-tmp/musicdata/train_tensors")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Scan only, don't modify")
    args = parser.parse_args()

    tensor_dir = Path(args.tensor_dir)
    pt_files = sorted(tensor_dir.glob("*.pt"))
    logger.info(f"Found {len(pt_files)} .pt files in {tensor_dir}")

    # Check which already have section_ids
    existing, missing = 0, 0
    sample = torch.load(str(pt_files[0]), map_location="cpu", weights_only=True)
    has_section_ids = "section_ids" in sample
    logger.info(f"Sample file has section_ids: {has_section_ids}")
    del sample

    if args.dry_run:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(process_file, str(f), dry_run=True): f for f in pt_files}
            total_sections = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    logger.info(result)
                    if "sections" in result:
                        total_sections += 1
            logger.info(f"Files with sections: {total_sections}/{len(pt_files)}")
        return

    # Process all files
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(process_file, str(f)): f for f in pt_files}
        ok, err = 0, 0
        for future in as_completed(futures):
            result = future.result()
            if result and result.startswith("ERROR"):
                logger.warning(result)
                err += 1
            else:
                ok += 1
        logger.info(f"Processed: {ok} OK, {err} errors out of {len(pt_files)}")


if __name__ == "__main__":
    main()
