# Paper Experiments: Marginal-Constrained Structured Conditioning

This directory contains the minimal evaluation pipeline for the paper on
**lyric-conditioned music generation with marginal-constrained transport
retrieval** (ACE-Step 1.5 DiT backbone).

## Directory Structure

```
scripts/paper_experiments/
  run_generation_variants.py    — Batch inference across model variants
  evaluate_lyrics.py            — Output-level lyric realisation metrics
  evaluate_internal_diagnostics.py — Internal transport diagnostic metrics
  plot_paper_figures.py         — Paper figures from metrics CSVs
  make_experiment_report.py     — Markdown + LaTeX report generator
  README.md                     — This file
```

## Variants

| Variant ID | Description | Sinkhorn | State Cost | Structural Units |
|-----------|-------------|----------|------------|-----------------|
| `baseline` | Original ACE-Step (no adapter) | No | N/A | N/A |
| `softmax_adapter` | Row-softmax over units | Row-softmax (no col constraint) | Yes | Yes |
| `transport_only` | Sinkhorn, qk_scale=0 | Yes | No | Yes |
| `full` | Full method | Yes | Yes | Yes |
| `no_structural_units` | Lyric-only units | Yes | Yes | No |

### Variant details

- **baseline**: Standard ACE-Step forward pass. No retrieval adapter, no
  PhaseMemory. Used as the primary baseline for all output-level metrics.

- **softmax_adapter**: ``RowSoftmaxRetrievalAdapter`` — mirrors the full
  adapter architecture (PM state, Q/K/V projections, RMS writer) but
  replaces Sinkhorn with a simple row-softmax:
  ``A_ij = softmax_j(L_ij); Pi_ij = nu_i * A_ij``.  No column marginal
  constraint.  Purpose: ablation control showing that *any* adapter is not
  enough — the marginal constraint matters.

- **transport_only**: ``TransportRetrievalAdapter`` with
  ``transport_qk_scale=0``.  The transport logit consists only of the
  position-based cost ``C = (p_audio - c_unit)^2 / sigma^2``.  No
  state-conditioned residual score.  Purpose: isolate the effect of
  Sinkhorn marginal constraints alone.

- **full**: ``TransportRetrievalAdapter`` with ``transport_qk_scale=1.0``.
  Combined logit ``L = -C + qk_scale * R`` where ``R`` is the
  PM-state-conditioned residual score.  Full Sinkhorn coupling.

- **no_structural_units**: Same as ``full``, but non-vocal units (INTRO,
  OUTRO, INSTRUMENTAL, auto-transition silences) are filtered out.
  Only lyric units remain.  Purpose: assess whether structural/silence
  units help with premature vocal entry.

## Pipeline

### 1. Generation

```bash
python scripts/paper_experiments/run_generation_variants.py \
    --prompts data/paper_prompts.jsonl \
    --output_dir outputs/paper_eval \
    --variants baseline,softmax_adapter,transport_only,full \
    --seeds 0,1 \
    --num_prompts 20 \
    --inference_steps 50 \
    --checkpoint_path /path/to/pm_retrieval.pt
```

Input ``prompts.jsonl`` format:
```json
{"prompt_id": "song_001", "lyrics": "[Verse] ...", "caption": "...", "duration": 180}
```

Output structure:
```
outputs/paper_eval/
  variant_name/
    prompt_id/
      seed_0/
        audio.wav
        metadata.json
        generation_config.json
        transport_diagnostics.npz   # (if applicable)
      seed_1/
        ...
```

### 2. Lyric Evaluation

```bash
python scripts/paper_experiments/evaluate_lyrics.py \
    --generation_dir outputs/paper_eval \
    --asr_backend external_json \
    --transcript_json data/asr_transcripts.json \
    --output_dir metrics/paper_eval
```

If ASR is unavailable, use ``--asr_backend external_json`` and provide
pre-computed transcripts:

```json
{"baseline/song_001/seed_0": "transcript text...", ...}
```

Or verbosely:
```json
[
    {"variant": "full", "prompt_id": "song_001", "seed": 0,
     "transcript": "transcript text..."}
]
```

### 3. Internal Diagnostics

```bash
python scripts/paper_experiments/evaluate_internal_diagnostics.py \
    --generation_dir outputs/paper_eval \
    --output_dir metrics/paper_eval
```

### 4. Figures

```bash
python scripts/paper_experiments/plot_paper_figures.py \
    --metrics_dir metrics/paper_eval \
    --figures_dir figures/paper_eval \
    --generation_dir outputs/paper_eval
```

### 5. Report

```bash
python scripts/paper_experiments/make_experiment_report.py \
    --metrics_dir metrics/paper_eval \
    --figures_dir figures/paper_eval \
    --output reports/paper_experiment_summary.md
```

## Metrics

### Output-level Metrics

All metrics are computed by aligning ASR transcript tokens with target
lyric tokens via LCS (longest common subsequence).  Chinese texts are
tokenised at the character level; English texts at the word level.

| Metric | Definition |
|--------|-----------|
| **Missing Rate** | Fraction of target lyric tokens not matched in transcript |
| **Repeat Rate** | Fraction of matched transcript tokens that match >1 target token |
| **Order Accuracy** | LCS length / total target tokens |
| **Order Error** | 1 - Order Accuracy |
| **Late Missing Rate** | Missing rate in the last third of target tokens |
| **Front/Middle/Late Missing** | Segment-level missing rates (0-33%, 33-66%, 66-100%) |
| **Front/Middle/Late Order Error** | Segment-level order errors |

### Internal Diagnostic Metrics

| Metric | Definition |
|--------|-----------|
| **Marginal Error** | ``L1(mu_hat - mu)`` where ``mu_hat = Pi.sum(axis=0)`` |
| **Cumulative Coverage Error** | ``E_cum(p) = sum_j |m_hat_j(p) - m_star_j(p)|`` over progress p |
| **State Cost Ratio** | L1 difference between full Pi and transport-only Pi (if paired) |

## Figures and Paper Sections

| Figure | Content | Likely Paper Section |
|--------|---------|---------------------|
| `fig3_segment_missing.pdf` | Front/Middle/Late Missing Rate | §3 or §4 (Main Results) |
| `main_output_metrics.pdf` | Main metrics bar chart | §4 (Main Results) |
| `cumulative_coverage_error.pdf` | E_cum(p) over progress | §4.2 (Internal Analysis) |
| `ablation_late_error.pdf` | Ablation on Late Missing | §4.3 (Ablation) |
| `transport_heatmap_case.pdf` (optional) | Pi heatmap | §4.2 (Qualitative) |

## Notes

### Without ASR

If you don't have an ASR system installed, use ``--asr_backend external_json``
and provide transcripts via JSON.  The evaluation script will skip ASR
inference and use the provided transcripts directly.

### Attention Staticity

Attention staticity diagnostics (inter-step attention cosine distance,
attention entropy) require hooking baseline cross-attention at layer 12
during generation.  This is **not** implemented in the current pipeline
because it requires changes to the frozen DiT backbone or an intrusive
forward hook.

If the paper needs to make claims about near-static attention in the
baseline:

1. Add a forward hook in ``run_generation_variants.py`` that saves
   ``cross_attn.attn`` (or the attention probabilities) for the baseline
   variant.
2. Compute ``D_attn(tau) = 1 - cosine(vec(A_{tau+1}), vec(A_{tau}))``
   for consecutive denoising steps.
3. Compute ``H(A_tau) = -sum(A_tau * log(A_tau + eps))`` for entropy.

If this analysis is absent, weaken the claim in the paper to:
> "Prior work suggests near-static attention may limit fine-grained
> lyric control in long-form generation; we adopt this as motivation
> rather than empirically verifying it in our setup."

### Paper Terminology

Recommended terminology (do not use overstated claims):

| ✅ Recommended | ❌ Avoid |
|---------------|---------|
| condition allocation | global lyric clock |
| structured condition coverage | guarantees monotonic alignment |
| coverage drift | solves all long-form drift |
| marginal-constrained coupling | prove attention is the root cause |
| phase-like denoising-state summary | |
| state-conditioned transport cost | |
| mitigates long-form lyric drift | |
| supports the formulation | |

### Dependencies

- Python >= 3.10
- torch >= 2.0
- numpy
- matplotlib (for plotting)
- ACE-Step environment with ``acestep`` package
- (optional) whisper / funasr for ASR

### Quick Start

```bash
# 1. Generation (requires GPU)
python scripts/paper_experiments/run_generation_variants.py \
    --prompts data/paper_prompts.jsonl --output_dir outputs/paper_eval \
    --variants baseline,softmax_adapter,transport_only,full --seeds 0,1

# 2. Evaluate with provided transcripts
python scripts/paper_experiments/evaluate_lyrics.py \
    --generation_dir outputs/paper_eval --asr_backend external_json \
    --transcript_json data/asr_transcripts.json --output_dir metrics/paper_eval

# 3. Internal diagnostics
python scripts/paper_experiments/evaluate_internal_diagnostics.py \
    --generation_dir outputs/paper_eval --output_dir metrics/paper_eval

# 4. Plot
python scripts/paper_experiments/plot_paper_figures.py \
    --metrics_dir metrics/paper_eval --figures_dir figures/paper_eval

# 5. Report
python scripts/paper_experiments/make_experiment_report.py \
    --metrics_dir metrics/paper_eval --figures_dir figures/paper_eval \
    --output reports/paper_experiment_summary.md
```

## Output Checklist

```
metrics/paper_eval/output_metrics_per_sample.csv
metrics/paper_eval/output_metrics_summary.csv
metrics/paper_eval/internal_cumulative_curve.csv
figures/paper_eval/fig3_segment_missing.pdf   + .png
figures/paper_eval/main_output_metrics.pdf    + .png
figures/paper_eval/cumulative_coverage_error.pdf + .png
figures/paper_eval/ablation_late_error.pdf    + .png
reports/paper_experiment_summary.md
```
