# OpenJev pilot results — 2026-09-29

The four LoRA runs completed successfully. Every selected adapter reloaded in a
fresh process and reproduced four saved dev predictions exactly. The original
OpenJev and Qwen APIs were both live after the supervisor completed. No adapter
was promoted to serving.

## Selected checkpoints

All four runs used the same 206 training and 52 development ML groups and the
same 2,000 / 501 replay NLI rows. Selection used development symmetric accuracy
subject to a replay retention gate.

| Model | Selected epoch | Symmetric dev correct | Raw swap consistency | Replay dev correct |
|---|---:|---:|---:|---:|
| Original 4B v5 | — | 33/52 | 29/52 | 481/501 |
| Seed 42, consistency 0 | 2 | 51/52 | 51/52 | 483/501 |
| Seed 42, consistency 0.1 | 2 | 51/52 | 52/52 | 481/501 |
| Seed 43, consistency 0 | 3 | 51/52 | 51/52 | 484/501 |
| Seed 43, consistency 0.1 | 2 | 51/52 | 51/52 | 484/501 |

All selected models made the same sole dev error: synthetic task seed 10313,
`ensemble_diversity`, where the empirical winner is B. This development set
was used for epoch selection.

## Historical canary diagnostic

We evaluated two earlier Crucible eval cohorts with the exact training NLI
template, both candidate orientations, legacy role-reference repair, and
symmetric aggregation. No candidate code hash or dataset group from these
39 pairs occurs in train or dev. The largest input was 1,667 tokens, below the
2,048-token training budget. Sources and per-pair scores are copied under
`artifacts/openjev_pilot_20260929/checks/historical_canaries/`.

| Model | Early canary | Later canary | Combined | Raw swap consistent |
|---|---:|---:|---:|---:|
| Original 4B v5 | 13/25 | 10/14 | 23/39 | 22/39 |
| Seed 42, consistency 0 | 21/25 | 14/14 | 35/39 | 34/39 |
| Seed 42, consistency 0.1 | 21/25 | 14/14 | 35/39 | 35/39 |
| Seed 43, consistency 0 | 21/25 | 14/14 | 35/39 | 36/39 |
| Seed 43, consistency 0.1 | 21/25 | 14/14 | 35/39 | 36/39 |

Relative to the original model, each adapter corrected 13 pair choices and
lost one. All four adapters chose the same candidates on all 39 pairs. Their
four errors are early canary seeds 1019, 1022, 1024 and 1039. Three are
`pathology_defense`; accuracy for that operator remains 1/4. The consistency
penalty did not change any candidate choice in this diagnostic.

The live original-model `/v1/compare` API made the same choice as offline
inference on all 39 pairs. Its symmetric scores differed by at most 0.020
because live inference batches four NLI inputs and the diagnostic processes
one input at a time in bf16.

These canaries were generated and examined during earlier development. The
later 14 had already informed the symmetry design. This diagnostic shows a
large improvement on historical tasks, but is **not an independent confirmation
of generalization**. The original collection is concentrated on synthetic
problems and one model family. A new frozen eval set with more varied datasets
and strong candidate pairs is required before promotion.

The serving API still uses the original 4B v5 checkpoint. Qwen was restored by
the supervisor. The complete server report is
`/data/openjev_adapt_20260929/pilot_report.json`; the local JSON diagnostics
contain source SHA-256 hashes and all per-pair outputs.
