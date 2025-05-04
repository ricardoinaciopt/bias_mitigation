"""
--mode   Which subset of files to evaluate. If omitted the script uses *both*.
--dir    Directory with the *.json metric files (default: abm_json).
--out    Path for the textual report (default: evaluation.txt).

Ranking methodology
-------------------
* **Fairness score**  - the *lower* the better:
      fairness = mean(|SPD|, |EOD|, |AOD|, |PrecisionDiff|)
* **Performance score** - the *higher* the better:
      performance = roc_auc + accuracy + precision + recall - log_loss
  (Log-loss is subtracted so that lower losses improve the score.)
* **Combined score** - trade-off between both criteria:
      combined = performance - fairness
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


FAIRNESS_KEYS = ("SPD", "EOD", "AOD", "PrecisionDiff")
PERFORMANCE_KEYS = ("roc_auc", "accuracy", "precision", "recall")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank ABM metric JSON files on fairness and performance"
    )
    parser.add_argument(
        "--mode",
        choices=("static", "dynamic", "both"),
        default="both",
        help="Subset of metric files to process",
    )
    parser.add_argument(
        "--dir",
        default="abm_json",
        help="Directory containing *.json metric files",
    )
    parser.add_argument(
        "--out",
        help="Destination text file for the evaluation report",
    )
    args = parser.parse_args()

    if args.out is None:
        eval_dir = Path("evaluation")
        args.out = eval_dir / f"evaluation_{args.mode}.txt"

    return args


def gather_files(directory: Path, mode: str) -> List[Path]:
    """Collect JSON file paths according to the chosen mode."""
    files = list(directory.glob("*.json"))
    if mode == "both":
        return files
    key = "metrics_static" if mode == "static" else "metrics_dynamic"
    return [p for p in files if p.name.startswith(key)]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def fairness_score(bias: Dict[str, float]) -> float:
    """Lower is better."""
    return sum(abs(bias[k]) for k in FAIRNESS_KEYS) / len(FAIRNESS_KEYS)


def performance_score(overall: Dict[str, float]) -> float:
    """Higher is better."""
    # negative log-loss rewards lower loss values
    return sum(overall[k] for k in PERFORMANCE_KEYS) - overall["log_loss"]


def combined_score(perf: float, fair: float) -> float:
    return perf - fair


def main() -> None:
    args = parse_args()

    directory = Path(args.dir).expanduser()
    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    metric_files = gather_files(directory, args.mode)
    if not metric_files:
        raise SystemExit("No metric JSON files were found - check the path/mode.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[float, float], List[Dict[str, Any]]] = defaultdict(list)

    for path in metric_files:
        data = load_json(path)
        rep = data["meta"]["representation_bias"]
        lbl = data["meta"]["label_bias"]

        fair = fairness_score(data["bias_metrics"])
        perf = performance_score(data["overall_metrics"])
        comb = combined_score(perf, fair)

        grouped[(rep, lbl)].append(
            {
                "file": path.name,
                "fair": fair,
                "perf": perf,
                "comb": comb,
            }
        )

    lines: List[str] = []
    lines.append(f"Evaluation mode: {args.mode}\n")

    best_overall = {
        "fair": min(
            (metric for grp in grouped.values() for metric in grp),
            key=lambda d: d["fair"],
        ),
        "perf": max(
            (metric for grp in grouped.values() for metric in grp),
            key=lambda d: d["perf"],
        ),
        "comb": max(
            (metric for grp in grouped.values() for metric in grp),
            key=lambda d: d["comb"],
        ),
    }

    for (rep, lbl), items in sorted(grouped.items()):
        lines.append(f"### rep = {rep}, lbl = {lbl}")

        ranked_fair = sorted(items, key=lambda d: d["fair"])
        ranked_perf = sorted(items, key=lambda d: d["perf"], reverse=True)
        ranked_comb = sorted(items, key=lambda d: d["comb"], reverse=True)

        def describe(title: str, ranked: List[Dict[str, Any]], key: str) -> None:
            lines.append(title)
            for idx, m in enumerate(ranked, 1):
                lines.append(f"  {idx:2d}. {m['file']:<60} {key} = {m[key]:.4f}")
            lines.append("")

        describe("Fairness ranking:", ranked_fair, "fair")
        describe("Performance ranking:", ranked_perf, "perf")
        describe("Combined ranking:", ranked_comb, "comb")

    lines.append("## Overall best models across all settings\n")
    lines.append(
        f"Most fair        : {best_overall['fair']['file']} (fair = {best_overall['fair']['fair']:.4f})"
    )
    lines.append(
        f"Best performance : {best_overall['perf']['file']} (perf = {best_overall['perf']['perf']:.4f})"
    )
    lines.append(
        f"Best combined    : {best_overall['comb']['file']} (comb = {best_overall['comb']['comb']:.4f})\n"
    )

    out_path.write_text("\n".join(lines))
    print(f"Report written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
