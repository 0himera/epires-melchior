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

The four runs finished successfully. Selected checkpoints and the historical
canary diagnostic are reported in [openjev-pilot-results-2026-09-29.md](openjev-pilot-results-2026-09-29.md).

## Throughput check

The original microbatch is 2 ML groups (8 sequences), followed by 8 replay
sequences, accumulated 4 times. The effective batch is 8 groups / 64 sequences;
the last step of each epoch contains 6 groups / 48 sequences.

`scripts/benchmark_openjev_training.py` compares identical groups and replay
examples, including a partial final batch. It checks loss and gradient agreement
with dropout disabled, then measures synchronized forward/backward times with
training dropout enabled, one warmup and three measured repetitions. Optimizer
state memory is included; optimizer updates, evaluation and checkpoint saves
are outside its timing. Exact training trajectories can still change with
microbatch size because dropout draws and floating point reductions change.

Disabling checkpointing failed even at microbatch 2: PyTorch had allocated
158.86 GiB when the GPU ran out of free memory. The paused training process and
serving process also occupied memory during this isolated compute benchmark.
This does not establish that the model alone exceeds the GPU's 192 GiB, but it
rules out using that configuration with these colocated processes.

Checkpointing remains enabled. `--micro-groups` and `--accumulation` control
batching; loss weights account for the actual number of groups in a partial
accumulation window. The supervisor can resume existing containers, including
verification containers, without rerunning or overwriting them. New batching
settings apply only to newly created training containers.

Server artifacts: `speed_benchmark.json`,
`speed_benchmark_no_checkpoint_failed.json`,
`speed_benchmark_no_checkpoint_failed.log`,
`speed_benchmark_gradient_failed.log` under the experiment directory.

Measured baseline: median 12.323 seconds per effective batch, 17.63 GiB peak
PyTorch allocation in the benchmark process. Microbatch 4 with checkpointing
failed the conservative gradient gate before timing: loss 0.756323 versus
0.756137, gradient cosine 0.984548, relative gradient error 0.187525 (dropout
disabled). This is not evidence of worse downstream quality, but does not
support treating the settings as a numerically interchangeable optimization.
Microbatch 8 was not reached. No acceleration has been demonstrated or applied.

The original seed-42 training container and supervisor were resumed unchanged;
the remaining pilot runs retain microbatch 2, accumulation 4 and checkpointing.
Further batching changes require investigating the gradient discrepancy and
checking training quality separately from this four-run comparison.
