# OpenJev 4B → 0.8B distillation pilot — 2026-09-30

Started at the user's request before the server's expected removal. This is a
finite compression experiment; its development scores do not establish an
improvement on independent tasks.

## Frozen recipe

- Teacher: `AlexWortega/openjev/qwen3.5-4b-nli-v5` plus our selected
  `s42_c00/epoch_2` LoRA adapter, SHA256
  `331f21571129c491afe49228ef2d7b0d14abcc9e898470e645de89cf8ab32bb0`.
- Student: `AlexWortega/openjev/qwen3.5-0.8b-nli-v2s-long`.
- Both public bases are pinned to repository revision
  `26de23c44b67586b4bea31c0ef2e016e3068ae66` in the server rescue metadata.
- Export the teacher's three raw logits for every existing ML/replay train/dev
  input. Input hashes, data hashes, adapter hash and logit-file hashes identify
  the teacher cache. Exact duplicate inputs reuse the same exported logits.
- One epoch: 206 ML groups / 824 NLI rows plus all 2,000 replay training rows.
  Both candidate orientations stay in the same optimizer window. Existing dev
  contains 52 ML groups / 208 NLI rows plus 501 replay rows. Group and exact input
  overlap between train/dev are rejected. No truncation or independent test use.
- Text LoRA rank 16, alpha 32, dropout .05; train the NLI score head; freeze vision.
  bf16, gradient checkpointing, AdamW lr `2e-5`, seed 42, gradient clipping 1.
  Microbatch two ML groups, accumulation four, 26 optimizer updates.
- Equal total ML/replay weights. Within each: `0.7 * T² * KL(teacher || student)`
  plus `0.3 * CE(gold)`, temperature `T=2`. Gold labels remain available to
  correct teacher errors. No additional positional consistency objective is
  imposed on this compression pilot.

## Save and evaluation

Before training, evaluate the unchanged student on the existing dev split.
After training, evaluate normal/reversed/symmetric ML accuracy, swap consistency,
replay NLI accuracy/loss, teacher/student class agreement and soft-distribution
KL. Save raw per-pair predictions. Reload the final adapter into a fresh base
and compare four dev predictions with the saved evaluation (tolerance `1e-4`).

Adapters are saved after update 1, every four updates and at the end. Checkpoint
directories are renamed atomically after saving. They contain adapters and step
metadata; optimizer state is not saved, so this is not exact training resumption.
The unchanged public student base is needed to load them. The tokenizer is saved
once at the run root; it is also available in the pinned public checkpoint.

The training loop stops at an optimizer boundary after **02:28 MSK**, saving and
evaluating a partial epoch if necessary. The host runner has a hard deadline of
**02:30 MSK**, writes status/logs, and restarts Qwen in `finally`. A hard timeout
can leave only earlier saved adapters. The separate local rescue process copies
new artifacts every 15 seconds and verifies stable output hashes when finished.

Server run: `/data/openjev_distill_20260930/`.

Private local rescue:
`/home/himera/server-backups/epires-129.212.176.219-20260930/data/openjev_distill_20260930/`.
Raw data and weights remain outside Git.

## Implementation and checks

- `scripts/distill_openjev.py`: teacher export, student training and reload check.
- `scripts/run_openjev_distillation.py`: isolated offline Docker stages, deadline,
  logs and Qwen restoration.
- `tests/test_openjev_distillation.py`: distillation gradient direction, logit
  offset invariance, zero gradient when distributions agree, gold correction,
  changed-source cache rejection and group leakage rejection.

Results are recorded in the private run's `student/summary.json`; do not infer
success merely from a container starting. No student is promoted to serving.
