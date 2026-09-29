# Fresh trained OpenJev results — 2026-09-29

All three collections completed and exited with code 0. The first Qwen canary
finished at 22:27:30 Moscow time, the additional Qwen canary at 22:46:39, and
the original-base comparison at 22:47:43. Both serving endpoints returned HTTP
200 afterwards. OpenJev still serves `s42_c00/epoch_2` with the pinned adapter
hash from the [frozen protocol](openjev-live-canary-2026-09-29.md).

## Same-pair comparison

The original base and the trained model were evaluated on the same saved pairs.
Only pairs with both candidates executing successfully and a bootstrap-confirmed
quality difference enter these accuracy denominators. Every other attempted
task is retained in the archive.

| Collection | Attempted tasks / comparisons | Decisive pairs | Original base correct | Trained correct |
|---|---:|---:|---:|---:|
| First Qwen canary | 60 tasks | 25 | 15/25 (60%) | 23/25 (92%) |
| Additional real-data Qwen canary | 24 tasks | 8 | 7/8 (87.5%) | 6/8 (75%) |
| Additional real-data sklearn portfolio | 72 comparisons on 24 tasks | 31 | 22/31 (71.0%) | 21/31 (67.7%) |

Across the two Qwen collections the trained model is correct on 29/33 pairs
versus 22/33 for the original base. Its raw choices are consistent under A/B
reversal on all 33 pairs; the portfolio has raw consistency on 26/31 pairs.
The primary decision averages both orientations, including inconsistent raw
views. There were no OpenJev inference failures or overlength exclusions.

The improvement in the first collection comes from synthetic tasks: 22/23
trained versus 14/23 original. Its only decisive real-data pairs are two vehicle
splits, with both models correct on 1/2. Banknote authentication and bank
marketing produced no decisive quality pair. The additional Qwen collection
has only eight decisive pairs spread across six datasets.

Consequently, these results support improvement on fresh synthetic Qwen
comparisons. They do **not** establish an improvement on diverse real datasets.
Four splits of one OpenML dataset are correlated and must not be treated as
four independent dataset groups.

## Specific real-data weaknesses

The additional Qwen collection introduced one regression relative to the base:
`cpu_small`, seed 92000, operator `inductive_bias` (empirical R² difference
0.2923). Both models also miss a house-prices pair, seed 92006 (`data_centric`).

The portfolio exposes a shared weakness on `cpu_small`: both models are correct
on only 1/9 decisive comparisons. On house prices the base is correct on 6/7
versus 5/7 for the adapter; the lost comparison is seed 3792413882,
`portfolio_hgb_vs_svm`. Both models are correct on all decisive portfolio pairs
from abalone (2), kc1 (2), phoneme (8), and cmc (3).

These are diagnostic observations on small, correlated subsets. The test
corpora remain held out; further training requires separate real-data groups.

## Collection reliability

- 84 Qwen tasks were attempted; 62 reached candidate execution.
- 22 tasks (26.2%) failed generation: 17 in the first collection and five in
  the additional one. Saved failed attempts comprise 23 timeouts at 1,200
  seconds and 21 completions ending at the 16,384-token limit.
- Another five pairs had a candidate execution failure; 24 successful pairs
  were inconclusive. This leaves 33 decisive Qwen pairs.
- All 144 fixed portfolio candidates executed successfully; its remaining
  41 comparisons were inconclusive.
- Candidate failures were invalid sklearn usage: pipeline sample-weight
  routing, unsupported histogram-boosting arguments, an incompatible Ridge
  solver, and an unsupported histogram-boosting loss.

The timeout and generation-budget failures reduce dataset coverage and waste
collection time. The configured `xhigh` reasoning and 16,384-token limit were
preserved; changing either was not part of this run. Generated candidate errors
were not repaired inside this frozen evaluation.

## Saved evidence

Server: `/data/openjev_live_20260929/`; local:
`artifacts/openjev_live_20260929/`. All arrays, manifests, transactional records,
reasoning traces, generated code, predictions, labels, errors and both-order
scores are retained. The completed outputs were copied locally; corpus hashes
from `base_rescore.json` and SQLite/JSONL record counts were checked.

`base_comparison.json` contains paired counts and decisions;
`base_rescore.json` contains original-model outputs and corpus hashes;
`run/` and `extras/` contain the raw collections. The supervisor state is
`complete`; no evaluation container remains running.
