"""
benchmarking/run.py — CLI entry point for the benchmarking pipeline.

Usage:
  # HuggingFace dataset (plain name)
  python benchmarking/run.py --dataset domenicrosati/TruthfulQA

  # HuggingFace — paste the load_dataset() snippet directly, it still works
  python benchmarking/run.py --dataset 'load_dataset("domenicrosati/TruthfulQA")'

  # CSV file
  python benchmarking/run.py --csv path/to/data.csv

  # Custom split and row cap
  python benchmarking/run.py --dataset imdb --split train[:1000] --max-rows 500

  # Custom output path
  python benchmarking/run.py --dataset imdb --output reports/my_report.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so pipeline.py can import from agents/
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_BENCH_DIR)
for _p in (_BENCH_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Input parsers ─────────────────────────────────────────────────────────────

def _parse_hf_name(raw: str) -> str:
    """
    Accept any of:
      domenicrosati/TruthfulQA
      load_dataset("domenicrosati/TruthfulQA")
      https://huggingface.co/datasets/domenicrosati/TruthfulQA
    and return just the dataset id.
    """
    m = re.search(r'load_dataset\s*\(\s*["\']([^"\']+)["\']', raw)
    if m:
        return m.group(1).strip()
    m = re.search(r'huggingface\.co/datasets/([^\s\'"]+)', raw)
    if m:
        return m.group(1).strip().rstrip("/")
    return raw.strip()


def _load_hf(raw_name: str, split: str):
    from datasets import load_dataset
    name = _parse_hf_name(raw_name)
    print(f"[BENCH] HuggingFace dataset : {name}  (split={split})")
    ds = load_dataset(name, split=split)
    # friendly short name for the report filename
    short = name.split("/")[-1]
    return ds, short


def _load_csv(path: str):
    import pandas as pd
    from datasets import Dataset
    print(f"[BENCH] CSV : {path}")
    df = pd.read_csv(path)
    return Dataset.from_pandas(df), Path(path).stem


# ── Terminal report printer ───────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GRADE_COLOR = {
    "A": "\033[92m",   # bright green
    "B": "\033[94m",   # blue
    "C": "\033[93m",   # yellow
    "D": "\033[91m",   # red
    "F": "\033[91m",   # red
}
_CAT_EMOJI = {
    "validity":    "✔ ",
    "fidelity":    "≈ ",
    "diversity":   "⊞ ",
    "readability": "📖",
    "other":       "·  ",
}


def _bar(score: float, width: int = 12) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _print_report(report: dict) -> None:
    summary = report.get("summary", {})
    metrics = [m for m in report.get("metrics", []) if not m.get("error")]
    skipped = [m for m in report.get("metrics", []) if m.get("error")]
    meta    = report.get("metadata", {})

    grade  = summary.get("grade", "?")
    score  = summary.get("overall_score") or 0.0
    gc     = _GRADE_COLOR.get(grade, "")

    # ── Hero: total aggregate score ───────────────────────────────────────────
    pct  = int(round(score * 100))
    bar  = _bar(score, width=20)
    print(f"\n{'═' * 68}")
    print(f"  {_BOLD}DATA QUALITY BENCHMARK  ·  {report['dataset_name'].upper()}{_RESET}")
    print(f"{'═' * 68}\n")
    print(f"  {_BOLD}TOTAL AGGREGATE SCORE{_RESET}")
    print(f"  {gc}{_BOLD}{score:.4f}  ({pct}%)   {bar}   Grade: {grade}{_RESET}\n")
    print(f"  {meta.get('row_count',0):,} rows × {meta.get('col_count',0)} cols  |  "
          f"primary column: {report.get('primary_column','—')}  |  "
          f"{summary.get('computed_count',0)} metrics computed")
    print()

    # ── Category breakdown (secondary) ───────────────────────────────────────
    by_cat = summary.get("by_category", {})
    if by_cat:
        print(f"  {'Category':<15}  {'Score':>6}  {'Bar':20}  Contribution")
        print(f"  {'─'*15}  {'─'*6}  {'─'*20}  {'─'*24}")
        for cat, info in sorted(by_cat.items(), key=lambda x: -x[1]["mean_score"]):
            em   = _CAT_EMOJI.get(cat, "·  ")
            bar2 = _bar(info["mean_score"], width=20)
            gc2  = ("\033[92m" if info["mean_score"] >= 0.75
                    else ("\033[93m" if info["mean_score"] >= 0.50 else "\033[91m"))
            names = ", ".join(m["name"].replace("_", " ") for m in info["metrics"][:3])
            if len(info["metrics"]) > 3:
                names += f" +{len(info['metrics'])-3}"
            print(f"  {em}{cat:<13}  {gc2}{info['mean_score']:>6.4f}{_RESET}  {bar2}  {names}")
        print()

    # ── Per-metric detail ─────────────────────────────────────────────────────
    if metrics:
        print(f"  {'Metric':<48}  {'Score':>6}  {'Anomalies':>9}")
        print(f"  {'─'*48}  {'─'*6}  {'─'*9}")
        for m in sorted(metrics, key=lambda x: -x.get("quality_score", 0)):
            name  = m["metric_name"][:47]
            sc    = m.get("quality_score", 0.0)
            anoms = m.get("anomalous_row_count", 0)
            gc2   = "\033[92m" if sc >= 0.75 else ("\033[93m" if sc >= 0.50 else "\033[91m")
            print(f"  {name:<48}  {gc2}{sc:>6.4f}{_RESET}  {anoms:>9}")

    if skipped:
        print(f"\n  Skipped ({len(skipped)}): "
              + ", ".join(m["metric_name"] for m in skipped))

    prob = summary.get("most_problematic_rows", [])
    if prob:
        row_ids = ", ".join(f"row {p['row']} (×{p['flagged_by']})" for p in prob[:5])
        print(f"\n  Most flagged rows  :  {row_ids}")

    print(f"\n{'═' * 68}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a dataset: universal + registry text quality metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset", "-d", metavar="HF_NAME",
        help="HuggingFace dataset id or load_dataset() snippet",
    )
    src.add_argument(
        "--csv", "-c", metavar="PATH",
        help="Path to a CSV file",
    )
    parser.add_argument(
        "--split", "-s", default="train[:500]",
        help="HuggingFace dataset split (default: train[:500])",
    )
    parser.add_argument(
        "--max-rows", "-n", type=int, default=500,
        help="Max rows to benchmark (default: 500)",
    )
    parser.add_argument(
        "--output", "-o", metavar="PATH",
        help="Output JSON path (default: benchmarking/reports/<name>_<timestamp>.json)",
    )
    args = parser.parse_args()

    # Load dataset
    if args.dataset:
        dataset, name = _load_hf(args.dataset, args.split)
    else:
        dataset, name = _load_csv(args.csv)

    # Run pipeline
    from pipeline import run_benchmark
    report = run_benchmark(dataset, dataset_name=name, max_rows=args.max_rows)

    # Print table
    _print_report(report)

    # Save JSON report
    reports_dir = os.path.join(_BENCH_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    if args.output:
        out_path = args.output
    else:
        ts        = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        out_path  = os.path.join(reports_dir, f"{safe_name}_{ts}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[BENCH] Report saved  →  {out_path}\n")


if __name__ == "__main__":
    main()
