# Trained OpenJev serving and fresh Qwen canary — 2026-09-29

## Serving change

The chosen adapter is `s42_c00/epoch_2`, selected by the existing development
protocol. Its weights SHA-256 is
`331f21571129c491afe49228ef2d7b0d14abcc9e898470e645de89cf8ab32bb0`.
The base is `qwen3.5-4b-nli-v5`; this deployment requires no further training.

`serve_openjev.py` loads the adapter with PEFT, reports its identity in health
and inference responses, and uses the same right padding, disabled cache and
single-input bf16 inference as the offline evaluation. The server keeps the
existing 8,192-token input limit and rejects longer inputs without truncation.
Training and earlier offline validation used at most 2,048 input tokens, so
new pairs above that length are a separate context-length transfer diagnostic.
The 16,384-token limit applies to Qwen generation, including reasoning.

`JevClient(mode='local')` now calls the local model server. Network failures
raise an error instead of silently substituting heuristic mock decisions.

## Frozen evaluation protocol

- 60 attempted tasks: 48 fresh synthetic realizations and four splits each of
  OpenML vehicle (54), banknote authentication (1462), and bank marketing (1461).
- New synthetic seeds: 90000–90071, excluding variants 4 and 5, which reuse
  built-in datasets. Real tasks use seeds 90100–90111 and split seeds 2718–2721.
- Fresh Qwen candidates, `enable_thinking=true`, `reasoning_effort=xhigh`,
  16,384 generation tokens, 32 workers, 1,200-second generation timeout.
- 60-second candidate execution timeout; maximum run duration three hours.
- Hidden test scoring in the parent process, paired bootstrap with 200 resamples,
  minimum effect 0.005. All errors and inconclusive pairs are retained.
- Primary judge result averages relative entailment across both A/B orders;
  raw order disagreement and choices are preserved.
- Require the adapter hash at startup and in every judge response. Record model
  identity alongside predictions. Never export this eval corpus into training.
- After collection, evaluate the **same frozen pairs** using the original base
  model. Group uncertainty by OpenML dataset, not by its four correlated splits.

Frozen pack SHA-256:
`ab7364e37a7ae2c4b1cead0dd638fef57f0e7287fcbc886e6e9881a112d4c405`.

## Saved data

Local artifacts: `artifacts/openjev_live_20260929/`.
Server artifacts: `/data/openjev_live_20260929/`.

`task_pack/` stores the protocol, profiles, preprocessing metadata, all train/test
arrays and their checksums. `run/` stores the manifest, transactional SQLite
records, candidate code, Qwen reasoning, predictions, hidden labels, inference
identity, judge scores in both orders, execution errors, exports and summary.
Deployment metadata and the serving smoke test are saved in the parent directory.

For new OpenML runs, preprocessing now removes explicit `id` columns only.
Continuous measurements with many distinct values remain. The previous
independent-test archive stays frozen as generated at commit `aa8c99b`.

This canary tests fresh candidate comparisons. It does not establish a gain in
the complete agent's iterative search under a fixed compute budget; that is the
next experiment after checking this collection and the base-model comparison.

## Additional real datasets

After the initial pack was frozen, six more datasets were downloaded and checked
against the official OpenML metadata and targets:

| Dataset | OpenML ID | Original rows / features | Metric |
|---|---:|---:|---|
| cpu_small | 562 | 8,192 / 12 | R² |
| house_prices | 42165 | 1,460 / 80 | R² |
| abalone | 183 | 4,177 / 8 | R² |
| kc1 | 1067 | 2,109 / 21 | F1 |
| phoneme | 1489 | 5,404 / 5 | ROC AUC |
| cmc | 23 | 1,473 / 9 | Accuracy |

The extra pack has four frozen splits per dataset (24 tasks), split seeds
4061–4064, task seeds 92000–92023, and SHA-256
`9dc9945e9e313437fdf80aaad1fb156b86e4006226dbf4c89e0b14538f4b4e53`.
OpenML sources and downloaded file checksums are in `dataset_search.json`.
Larger datasets are deterministically subsampled to 2,500 rows before the split.
Abalone's target is the numeric ring count, not a 28-class classification.

Two additional runs use these inputs: a fixed six-algorithm portfolio with
144 executions / 72 pairs, and a queued 24-task Qwen canary after the first
Qwen canary exits. Generation concurrency stays at 32 workers. The portfolio
runs with four CPU workers and scores the selected adapter through serving.
Each run preserves its own corpus; outputs are under `extras/portfolio/` and
`extras/qwen/`. Checkpoint and preprocessing are fixed before either run.

A detached supervisor waits for both Qwen runs and the portfolio, then scores
the original base model on their saved decisive pairs in a separate process.
The comparison uses the validated 2,048-token budget; longer inputs are recorded
as exclusions, while their original trained-serving scores remain archived.
Paired base/adapted counts are saved by cohort in `base_comparison.json`.
The trained serving container stays available throughout this comparison.
