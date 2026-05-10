"""Tests for JSONL resume prefix counting in caption repair script."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from caption import _count_processed_prefix_jsonl, _has_processed_meta, process_jsonl_stream


class CaptionResumeTests(unittest.TestCase):
    """Behavior tests for resume boundary detection."""

    def test_has_processed_meta_treats_error_as_unfinished(self) -> None:
        """It should not mark ``status=error`` records as completed."""
        self.assertTrue(_has_processed_meta({"caption_tag_fix_meta": {"status": "ok"}}))
        self.assertTrue(_has_processed_meta({"caption_tag_fix_meta": {"status": "skipped"}}))
        self.assertFalse(_has_processed_meta({"caption_tag_fix_meta": {"status": "error"}}))

    def test_count_processed_prefix_stops_on_error_status(self) -> None:
        """It should stop contiguous processed counting when an error record appears."""
        records = [
            {"id": 1, "caption_tag_fix_meta": {"status": "ok"}},
            {"id": 2, "caption_tag_fix_meta": {"status": "error"}},
            {"id": 3, "caption_tag_fix_meta": {"status": "ok"}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "out.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for obj in records:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            self.assertEqual(1, _count_processed_prefix_jsonl(jsonl_path))

    def test_resume_reuses_finished_records_after_earlier_error(self) -> None:
        """It should skip finished records even when an earlier record had error status."""
        input_records = [
            {"audio_path": "a1.wav", "caption": "pop, 120 bpm"},
            {"audio_path": "a2.wav", "caption": "rock, 130 bpm"},
            {"audio_path": "a3.wav", "caption": "jazz, 110 bpm"},
        ]
        existing_output_records = [
            {
                "audio_path": "a1.wav",
                "caption": "kept_line_1",
                "caption_tag_fix_meta": {"status": "ok"},
            },
            {
                "audio_path": "a2.wav",
                "caption": "line_2_old_error",
                "caption_tag_fix_meta": {"status": "error"},
            },
            {
                "audio_path": "a3.wav",
                "caption": "kept_line_3",
                "caption_tag_fix_meta": {"status": "ok"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.jsonl"
            output_path = Path(tmpdir) / "out.jsonl"

            with input_path.open("w", encoding="utf-8") as f:
                for obj in input_records:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            with output_path.open("w", encoding="utf-8") as f:
                for obj in existing_output_records:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            args = argparse.Namespace(
                input=str(input_path),
                output=str(output_path),
                audio_root=str(Path(tmpdir) / "audios"),
                audio_field="",
                caption_field="",
                write_field="",
                model="mock-model",
                api_key="mock-key",
                base_url="https://example.com",
                timeout=1,
                segment_count=1,
                segment_duration=1.0,
                request_retries=1,
                retry_delay=0.0,
                max_items=0,
                start_from=1,
                save_every=1,
                resume=True,
                check_api=False,
                api_check_only=False,
                force_overwrite=False,
            )

            process_jsonl_stream(args)

            out_records = []
            with output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out_records.append(json.loads(line))

            self.assertEqual("kept_line_1", out_records[0]["caption"])
            self.assertEqual("ok", out_records[0]["caption_tag_fix_meta"]["status"])

            self.assertEqual("skipped", out_records[1]["caption_tag_fix_meta"]["status"])
            self.assertEqual("audio_file_missing", out_records[1]["caption_tag_fix_meta"]["reason"])

            self.assertEqual("kept_line_3", out_records[2]["caption"])
            self.assertEqual("ok", out_records[2]["caption_tag_fix_meta"]["status"])


if __name__ == "__main__":
    unittest.main()
