"""
Lyrics Structure Parser for Section-RoPE Offset.

Parses raw lyrics text with section annotations ([Verse], [Chorus], etc.)
into structured outputs: section_type, section_index, repeat_group.

Does NOT modify the original lyrics text — only produces side-channel IDs
for Section-RoPE lyric structure embeddings.

Section types are normalized to a controlled vocabulary:
    INTRO, VERSE, PRE_CHORUS, CHORUS, BRIDGE, OUTRO,
    INSTRUMENTAL, HOOK, UNKNOWN
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalised section vocabulary
# ---------------------------------------------------------------------------

SECTION_VOCAB = {
    "unknown": 0,
    "intro": 1,
    "verse": 2,
    "pre-chorus": 3,
    "pre_chorus": 3,
    "chorus": 4,
    "bridge": 5,
    "outro": 6,
    "instrumental": 7,
    "hook": 4,
}
SECTION_NAMES = {v: k for k, v in SECTION_VOCAB.items()}
NUM_SECTION_TYPES = max(SECTION_VOCAB.values()) + 1  # 8


def compute_token_weights(section_ids: torch.Tensor) -> torch.Tensor:
    """Token-aware phase weighting based on section segment positions.

    Within each contiguous section segment:
      - UNKNOWN (ID=0): weight 0
      - First 15% of tokens: weight 1.0
      - Middle 70%:          weight 0.5
      - Last 15%:            weight 0.3

    Args:
        section_ids: [B, L] integer section type IDs.

    Returns:
        weights: [B, L] float32 weights in [0, 1].
    """
    B, L = section_ids.shape
    weights = torch.zeros(B, L, dtype=torch.float32, device=section_ids.device)

    for b in range(B):
        ids = section_ids[b]
        # Find contiguous-segment boundaries
        boundaries = [0]
        for i in range(1, L):
            if ids[i] != ids[i - 1]:
                boundaries.append(i)
        boundaries.append(L)

        for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
            seg_len = seg_end - seg_start
            if seg_len == 0:
                continue
            if ids[seg_start].item() == 0:  # SECTION_UNKNOWN
                weights[b, seg_start:seg_end] = 0.0
                continue

            head_len = max(1, int(seg_len * 0.15))
            tail_len = max(1, int(seg_len * 0.15))
            # Guard against overlap on very short segments
            middle_start = seg_start + head_len
            middle_end = max(middle_start, seg_end - tail_len)

            weights[b, seg_start:middle_start] = 1.0
            weights[b, middle_start:middle_end] = 0.5
            weights[b, middle_end:seg_end] = 0.3

    return weights


def _normalize_section_name(raw: str) -> str:
    """Normalise a raw section label to the controlled vocabulary.

    e.g. "[Verse 1]" -> "verse", "[Pre-Chorus]" -> "pre_chorus"
    """
    s = raw.strip().lower().strip("[]")
    # Remove trailing numbers: "verse1" -> "verse"
    s = re.sub(r"\d+$", "", s).strip()
    # Trim whitespace / hyphens
    s = s.replace("-", "_").replace(" ", "_")
    if s in SECTION_VOCAB:
        return s
    # Fuzzy fallback
    for key in SECTION_VOCAB:
        if key in s or s in key:
            return key
    return "unknown"


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

@dataclass
class LyricsStructureOutput:
    """Structure information for one song.

    Fields are 1D integer tensors/arrays aligned to lyric chunks:
        section_type_ids:   [C]  — 0..8 in SECTION_VOCAB
        section_index_ids:  [C]  — 0-based occurrence index within type
        repeat_group_ids:   [C]  — 0-based repeat group identifier
        num_sections:       int  — raw number of distinct sections
        num_unique_chunks:  int  — number of lyric chunks (C)
    """
    section_type_ids:      torch.Tensor  # [C] int64
    section_index_ids:     torch.Tensor  # [C] int64
    repeat_group_ids:      torch.Tensor  # [C] int64
    num_sections:          int = 0
    num_unique_chunks:     int = 0
    raw_sections:          List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class LyricsStructureParser:
    """Parses raw lyrics text into structured section annotations.

    Usage:
        parser = LyricsStructureParser()
        output = parser.parse(lyrics_text, num_chunks=16)
        # output.section_type_ids.shape == [16]
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: Fraction of shared words needed for two
                sections to be considered the same ``repeat_group``.
        """
        self.similarity_threshold = similarity_threshold

    def parse(
        self,
        lyrics_text: str,
        num_chunks: int = 16,
    ) -> LyricsStructureOutput:
        """Parse lyrics and produce chunk-level structure IDs.

        Args:
            lyrics_text: Raw lyrics string (with [Verse]/[Chorus] markers).
            num_chunks: Number of lyric chunks to produce IDs for.

        Returns:
            LyricsStructureOutput with aligned tensors.
        """
        if not lyrics_text or not lyrics_text.strip():
            return self._empty_output(num_chunks)

        # ---- Step 1: Extract labelled sections ----
        lines = lyrics_text.split("\n")
        section_boundaries: List[Tuple[int, str]] = []  # (line_idx, label)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                label = stripped
                normalized = _normalize_section_name(label)
                # Ignore duplicate consecutive labels (e.g. blank [Intro] lines)
                if not section_boundaries or section_boundaries[-1][1] != normalized:
                    section_boundaries.append((i, normalized))

        if not section_boundaries:
            # No labels found — whole song is "unknown"
            return self._empty_output(num_chunks)

        # ---- Step 2: Assign section_type and section_index per section ----
        type_counts: Dict[str, int] = {}
        section_data: List[Tuple[int, str, int]] = []  # (type_id, index_within_type, label)

        for _, norm_label in section_boundaries:
            type_id = SECTION_VOCAB.get(norm_label, SECTION_VOCAB["unknown"])
            count = type_counts.get(norm_label, 0)
            type_counts[norm_label] = count + 1
            section_data.append((type_id, count, norm_label))

        # ---- Step 3: Assign repeat_group based on text similarity ----
        repeat_groups: List[int] = []
        section_texts: List[str] = []

        # Collect text for each section
        for i, (line_idx, _) in enumerate(section_boundaries):
            start = line_idx + 1
            end = section_boundaries[i + 1][0] if i + 1 < len(section_boundaries) else len(lines)
            text = " ".join(
                lines[j].strip()
                for j in range(start, end)
                if lines[j].strip() and not (lines[j].strip().startswith("[") and lines[j].strip().endswith("]"))
            )
            section_texts.append(text)

        used_groups: List[str] = []  # each entry is a canonical text signature
        for text in section_texts:
            normalized_text = self._normalize_text(text)
            matched = False
            for gi, candidate in enumerate(used_groups):
                sim = self._text_similarity(normalized_text, candidate)
                if sim >= self.similarity_threshold:
                    # Same repeat group
                    repeat_groups.append(gi)
                    matched = True
                    break
            if not matched:
                # New repeat group
                repeat_groups.append(len(used_groups))
                used_groups.append(normalized_text)

        # ---- Step 4: Project to chunk-level IDs ----
        # Each chunk corresponds to a range of section tokens
        num_raw_sections = len(section_data)
        C = num_chunks

        type_ids = torch.zeros(C, dtype=torch.long)
        index_ids = torch.zeros(C, dtype=torch.long)
        group_ids = torch.zeros(C, dtype=torch.long)

        if num_raw_sections > 0:
            for c in range(C):
                # Map chunk c to a section
                sec_idx = int(c * num_raw_sections / C)
                sec_idx = min(sec_idx, num_raw_sections - 1)
                type_ids[c] = section_data[sec_idx][0]
                index_ids[c] = section_data[sec_idx][1]
                group_ids[c] = repeat_groups[sec_idx]

        return LyricsStructureOutput(
            section_type_ids=type_ids,
            section_index_ids=index_ids,
            repeat_group_ids=group_ids,
            num_sections=num_raw_sections,
            num_unique_chunks=C,
            raw_sections=[s[1] for s in section_data],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_output(self, num_chunks: int) -> LyricsStructureOutput:
        """Return all-unknown output when no structure is detected."""
        unknown_id = SECTION_VOCAB["unknown"]
        return LyricsStructureOutput(
            section_type_ids=torch.full((num_chunks,), unknown_id, dtype=torch.long),
            section_index_ids=torch.zeros(num_chunks, dtype=torch.long),
            repeat_group_ids=torch.zeros(num_chunks, dtype=torch.long),
            num_sections=0,
            num_unique_chunks=num_chunks,
            raw_sections=[],
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lower-case, remove punctuation, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _text_similarity(t1: str, t2: str) -> float:
        """Jaccard similarity of word sets between two normalised texts."""
        w1 = set(t1.split())
        w2 = set(t2.split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)
