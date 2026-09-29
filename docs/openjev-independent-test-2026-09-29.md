# OpenJev independent portfolio test — 2026-09-29

## Protocol and data

The protocol and candidate implementations were frozen before any model was
scored. The benchmark has 82 tasks: 64 new synthetic realizations from Crucible's
existing generator and three train/test splits of each of six previously unused
OpenML datasets (31 credit-g, 36 segment, 44 spambase, 1590 adult, 43257
wine_quality, 45026 heloc). For each task, a predetermined graph compares
three disjoint pairs from six fixed sklearn algorithms. Candidate orientation
is determined by a hash of the task ID and pair number. Test labels are withheld
from candidate subprocesses. The original paired-bootstrap arbiter uses 200
resamples and a 0.005 minimum metric difference.

All 492 candidate executions succeeded. Of 246 comparisons, 152 had a decisive
paired interval (74 A, 78 B) and 94 were marked `UNCERTAIN`; the latter were
preserved but excluded from accuracy scoring by the frozen protocol. The 152
decisive pairs span 62 synthetic tasks and six OpenML datasets (68 independent
clusters for analysis). The corpus SHA-256 is
`f8711582a98ba1242e033a0d34562b95760f61903fa185f61783209458502bc6`.
There were no dataset-group or exact candidate-code overlaps with ML train,
dev, or either historical canary cohort; source hashes and counts are in
`overlap_check.json`.

The complete local artifact is
`artifacts/openjev_independent_20260929/` and the server copy is
`/data/openjev_independent_20260929/`. The `protocol.json` records the fixed
construction, `arrays/*.npz` contains all inputs and hidden labels,
`candidates.jsonl` contains every implementation, prediction and execution
result, `evaluations.jsonl` contains all 246 labeled comparisons, and
`holdout.jsonl` contains the 152 scored pairs. `manifest.json` hashes the 86
original files. `scores/*.json` contains every raw NLI score in both orders,
plus model decisions; `scores/analysis.json` contains comparisons and task
cluster bootstrap intervals. The server archive's 86 original file hashes
match the local manifest.

## Results

No input was truncated; the longest was 302 tokens against a 2,048-token
budget. All runs used the fixed development-selected checkpoint for each
adapter. Symmetric accuracy and raw order consistency were:

| Model | All correct | Synthetic correct | OpenML correct | Order consistent |
|---|---:|---:|---:|---:|
| Original 4B v5 | 103/152 | 83/130 | 20/22 | 91/152 |
| Seed 42, consistency 0 | 113/152 | 92/130 | 21/22 | 145/152 |
| Seed 42, consistency 0.1 | 113/152 | 92/130 | 21/22 | 143/152 |
| Seed 43, consistency 0 | 113/152 | 92/130 | 21/22 | 144/152 |
| Seed 43, consistency 0.1 | 113/152 | 92/130 | 21/22 | 144/152 |

All four adapters made the same 152 symmetric decisions. Relative to the
original, each fixed 13 choices and lost three, a +6.58 percentage-point
difference. The 95% bootstrap interval from resampling entire tasks, with
OpenML splits clustered by source dataset, is +3.07 to +10.56 points. This
interval describes variation across the benchmark's tasks. For OpenML alone,
the difference is one corrected choice with no regressions, and the six-dataset
cluster interval includes zero. Baseline OpenML accuracy was already 20/22;
this is a small sample with a ceiling effect.

This test is independent of model tuning, but its hand-written sklearn
candidate distribution differs from Qwen-generated Crucible pairs. The synthetic
tasks are new realizations of generator families used before. These results
support improvement in this fixed portfolio setting and order robustness; they
do not by themselves establish the same improvement in fresh Qwen self-play or
justify changing the live service. The live service continues to use the
original checkpoint.

The fixed preprocessing also removed numeric columns with more than 80% unique
values: `credit_amount` from credit-g and `saturation-mean`/`hue-mean` from
segment, in addition to the `id` column from wine_quality. The full input
arrays and dropped-column lists are archived. Thus the OpenML results describe
these frozen feature subsets, not each dataset's full original feature set.

To reproduce: run `scripts/build_openjev_independent_test.py` with `uv run`
in the project environment, then `scripts/run_openjev_independent_server.py`
on the GPU server after copying the frozen corpus and evaluator, and finally
`scripts/analyze_openjev_independent.py` on the scored archive. Rebuilding from
OpenML may fetch a revised source; use the archived arrays for an exact replay.
