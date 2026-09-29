# OpenJev LoRA pilot — 2026-09-29

## Implemented

- Candidate-role prose is normalized before shuffling. Historical records use
  `generation_swapped` to recover original roles; code is preserved unchanged.
- Shared symmetric scoring returns uncertainty at numerical ties and retains
  the raw scores from both orientations.
- Server `/v1/compare` accepts `{profile, pair, generation_swapped?}` and scores
  four NLI sequences in one batch. `/v1/systemone` keeps its existing interface.
- The server still serves the original OpenJev v5 checkpoint. Training adapters
  are written separately and are never promoted automatically.

## Frozen data

Source: audited 258 pairs, with role-reference repair applied.

| Partition | ML groups | ML NLI rows | Original-corpus NLI rows |
|---|---:|---:|---:|
| Train | 206 | 824 | 2,000 |
| Dev | 52 | 208 | 501 |

Splitting precedes augmentation. Both orientations and both claims remain in the
same dataset group. Replay train/dev are separated by normalized premise hash.
Replay is sampled from the pinned OpenJev source dataset revision
`98ddc2bba16930bc975b711bb3bca269dce4459c`, files `text.jsonl.gz` and
`ifcomplex.jsonl.gz`. All three NLI labels are present. No sampled row exceeded
the 2,048-token budget; no truncation was performed.

Local data: `artifacts/openjev_pilot_20260929/data/`.
Server experiment: `/data/openjev_adapt_20260929/`.

## Measured initial baseline

| Metric | Original model |
|---|---:|
| Dev normal-order accuracy | 33/52 = 63.46% |
| Dev reversed-order accuracy | 32/52 = 61.54% |
| Raw order consistency | 29/52 = 55.77% |
| Symmetric accuracy | 33/52 = 63.46% |
| Always A / always B | 50% / 50% |
| Replay dev accuracy | 481/501 = 96.01% |

The real `/v1/compare` endpoint was tested with repaired seed 10081 and its reversed
presentation. The selected identity and score were identical (0.8480926076);
only the displayed A/B label changed.

## Finite training experiment

Four runs: seeds 42 and 43, each with consistency weight 0 and 0.1.
Each run has three epochs, saved/evaluated separately. First run:
`openjev-lora-pilot-s42-c01-v2` (the earlier startup failed on an unnecessary
sklearn import; the shared package import has been corrected).

Text-only LoRA rank 16, alpha 32, dropout .05, plus the existing `score` head;
32,472,576 trainable parameters. Vision is frozen. bf16, lr=2e-5, effective
8 ML groups per optimizer step, equal ML/replay CE weights, gradient clipping 1.
Both orientations are included in the same step. There are 78 optimizer steps
per run; the first steps took approximately 13–17 seconds each.

Checkpoint selection is fixed in advance: highest symmetric dev accuracy among
epochs whose replay accuracy drops by at most 2 percentage points from baseline;
ties use raw swap consistency, then the earlier epoch. Selected adapters are
reloaded in a fresh process and compared to their saved dev predictions.

The host supervisor writes `/data/openjev_adapt_20260929/pilot_report.json` after
each run and on completion/failure. It restarts `vllm-qwen` after the finite
experiment. The generator is temporarily stopped to release GPU memory;
the original OpenJev API remains available.

## Interpretation

All scores here are development results. The 52 dev pairs are used for checkpoint
selection. Existing 14 canary pairs have also been inspected during development.
Neither is a fresh independent test. Promotion requires a separate frozen test;
this experiment does not start another long data-generation run or replace the
base model automatically.

The original data's concentration on synthetic tasks and HistGradientBoosting
remains. Role repair and symmetric aggregation address presentation bias; they
do not broaden the training distribution.

## Commands and artifacts

- `scripts/prepare_openjev_adaptation.py`: grouped rotations and replay sampling.
- `scripts/train_openjev_adapter.py`: grouped CE/consistency training and dev evaluation.
- `scripts/verify_openjev_adapter.py`: adapter reload verification.
- `scripts/supervise_openjev_pilot.py`: finite experiment and generator restoration.
- `runs/*/baseline.json`, `epoch_*/evaluation.json`, `summary.json`,
  `reload_verification.json`: server results, including per-pair predictions.

Training results are not yet filled into this document; consult the experiment
report for actual completion state rather than treating this setup record as a
successful training result.
