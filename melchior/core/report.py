"""Report generator — creates audit monographs, calibration metrics, and reproducible artifacts."""

from __future__ import annotations

from pathlib import Path
from melchior.core.task import Task
from melchior.core.journal import Journal


def generate_report(
    journal: Journal, task: Task, output_dir: Path | str = "artifacts"
) -> str:
    """Generate final research report markdown and best solution script."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = journal.best_record(task.direction)
    best_val_str = f"{best.actual_metric:.4f}" if best and best.actual_metric is not None else "N/A"
    best_tech = best.technique if best else "None"
    best_hash = best.code_hash if best else "N/A"

    cal_stats = journal.calibration_stats()

    # Save best solution code
    if best and best.code:
        (out_dir / "best_solution.py").write_text(best.code, encoding="utf-8")

    lines = [
        f"# Melchior Research Monograph — {task.name}",
        "",
        "## Executive Summary",
        f"- **Task Goal**: {task.goal}",
        f"- **Domain**: {task.domain}",
        f"- **Target Metric**: `{task.metric}` ({task.direction})",
        f"- **Best Result Achieved**: **`{best_val_str}`** via `{best_tech}`",
        f"- **Best Code Hash**: `{best_hash}` (saved to `best_solution.py`)",
        f"- **Total Trials**: {len(journal.records)}",
        "",
        "## Dual-Brain Calibration Statistics (System 1 vs System 2)",
        "Melchior monitors self-calibration by comparing predicted vs empirical metrics:",
        f"- **Mean Calibration Bias**: `{cal_stats.get('mean_error', 0.0):+.4f}` "
        f"({'overestimating' if cal_stats.get('mean_error', 0) > 0 else 'underestimating'})",
        f"- **Mean Absolute Error (MAE)**: `{cal_stats.get('mae', 0.0):.4f}`",
        f"- **Evaluated Pairs**: {cal_stats.get('count', 0)}",
        "",
        "## Experiment Ledger",
        "| Trial | Strategy | Technique | Predicted | Actual | Δ (Error) | Status | Time |",
        "|:---|:---|:---|:---|:---|:---|:---|:---|",
    ]

    for r in journal.records:
        act = f"{r.actual_metric:.4f}" if r.actual_metric is not None else "-"
        delta = f"{r.calibration_error:+.4f}" if r.calibration_error is not None else "-"
        lines.append(
            f"| {r.trial_idx} | {r.strategy} | {r.technique} | {r.predicted_metric:.4f} | "
            f"**{act}** | {delta} | {r.status} | {r.wall_time_s:.2f}s |"
        )

    lines.extend([
        "",
        "## Best Model Implementation",
        "```python",
        best.code if best else "# No successful trial",
        "```",
        "",
        "---",
        "*Generated autonomously by Melchior dual-brain auto-research harness.*",
    ])

    report_text = "\n".join(lines)
    report_file = out_dir / "report.md"
    report_file.write_text(report_text, encoding="utf-8")

    return report_text
