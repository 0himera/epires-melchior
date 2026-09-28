# OpenJev data contract audit (2026-09-29)

Primary sources:
- https://huggingface.co/datasets/AlexWortega/openjev-data
- https://huggingface.co/datasets/AlexWortega/openjev-data/blob/main/README.md
- https://huggingface.co/datasets/AlexWortega/openjev-data/blob/main/composition.json
- https://huggingface.co/AlexWortega/openjev/blob/main/code/openjev_decide.py

The card reports 2,346,729 rows across 12 compressed JSONL files. These are
source corpora, not the exact final checkpoint training split. `composition.json`
reports 736,284 training and 2,000 validation examples for the v5 mixture,
including a benchmark panel deliberately excluded from this published dataset.
That exclusion does not undo the v5 checkpoint's exposure to benchmark test sets.
Do not use the listed MMLU/ARC/GSM8K/etc. panel as independent evaluation of v5.

Downloaded and scanned every row of `jevfmt.jsonl.gz` (102,586),
`ifcomplex.jsonl.gz` (382,331), and `distill.jsonl.gz` (27,310).
All have exactly premise, hypothesis, label, source, image. Labels are
0=contradiction, 1=entailment, 2=neutral. These three files contain only 0 and 1;
they do not turn uncertain comparisons into neutral examples. Image is an empty
string in the inspected text rows, despite the card describing null.

## Decision semantics

OpenJev explicitly learns choice, ordinal, probability, policy, adequacy and
best-response decisions, beyond literal textual entailment. In ifc_best_of,
premise contains the request and all three responses; hypothesis asserts that
a particular option satisfies the rubric. A winning option gets 1 and a losing
option gets 0. Therefore empirical ML preferences can use this scheme too,
provided the proposition actually asserts a choice under a defined criterion.
A free-form claim about an algorithm is not equivalent to that proposition.

The deployed wrapper's canonical hypothesis is:

    The answer to "{instructions}" is {label}: {criterion}

Training/export and serving must use the same state, question, criteria and
hypothesis template. Our comparison state includes both candidate rationales
and implementations, with no measured scores, errors or winner. Empirical
labels represent predictions of performance on independent test data, not
logical proofs of general algorithm superiority.

The distillation file repeats identical premise/hypothesis pairs with different
binary labels: teacher votes encode a distribution. For example, one inspected
routing option has [1, 1, 0, 1]. Removing these as duplicate/conflicting annotations
would destroy the intended weighting. Holdout splits must group by scenario,
not randomly divide option/vote rows. Teacher votes are not guaranteed truth;
individual inspected scenarios also contain ambiguous or inconsistent wording.

## Consequences for Crucible

- Export exactly the five core columns; keep execution/provenance in raw records.
- Share typed-decision formatting between training and inference.
- Failed execution is separate from performance preference. No artificial delta=1.
- Emit quality labels only for valid paired predictions and a supported margin.
- Keep ties/uncertainty in raw records; do not automatically label them neutral.
- Compute metrics outside generated code with hidden test labels.
- Use independent task instances and disjoint built-in datasets for train/eval;
  all operators use the same evaluation seeds. This is an in-family benchmark,
  not evidence of generalization to all real-world datasets.

The HF automatic viewer currently fails because it tries to cast composition
metadata as examples. Load explicit `data_files`, not all repository JSON files.
