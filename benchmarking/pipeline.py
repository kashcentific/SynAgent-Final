"""
benchmarking/pipeline.py

Lightweight benchmarking pipeline — no LLM calls, no research agent, no evaluator.

Stages:
  1. Load & cap dataset
  2. Extract column metadata (deterministic)
  3. Detect text columns
  4. Universal Text Metrics  (validity / fidelity / diversity)
  5. Tool Registry metrics   (readability / fluency via NLTK, HF evaluate, deepeval)
  6. Aggregate & grade

Returns a self-contained report dict that run.py saves as JSON.
"""

import datetime
import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.universal_text_metrics import UniversalTextMetrics
from agents.tool_registry import ToolRegistry
from config import MAX_ROWS_FOR_EVAL, CATEGORICAL_RATIO


# ── Column helpers ────────────────────────────────────────────────────────────

def _infer_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    dtype_str = str(series.dtype).lower()
    is_str = (
        series.dtype == object
        or "string" in dtype_str
        or "large_string" in dtype_str
        or "utf8" in dtype_str
    )
    if is_str:
        ratio = series.nunique(dropna=True) / max(len(series), 1)
        return "categorical" if ratio < CATEGORICAL_RATIO else "text"
    return "unknown"


def _detect_text_cols(df: pd.DataFrame, min_avg_len: int = 15) -> List[str]:
    text_cols = []
    for col in df.columns:
        s = df[col]
        if (pd.api.types.is_numeric_dtype(s)
                or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s)):
            continue
        try:
            avg = s.dropna().astype(str).str.len().mean()
            if pd.notna(avg) and float(avg) > min_avg_len:
                text_cols.append(col)
        except Exception:
            pass
    return text_cols


def _extract_col_stats(df: pd.DataFrame) -> List[Dict]:
    n = len(df)
    cols = []
    for col in df.columns:
        s = df[col]
        dtype = _infer_dtype(s)
        info: Dict[str, Any] = {
            "name":          col,
            "dtype":         dtype,
            "null_count":    int(s.isnull().sum()),
            "null_pct":      round(s.isnull().mean() * 100, 2),
            "unique_count":  int(s.nunique(dropna=True)),
        }
        if dtype == "numerical":
            arr = s.dropna().to_numpy(dtype=float, na_value=float("nan"))
            arr = arr[~np.isnan(arr)]
            if len(arr):
                info["stats"] = {
                    "min":      round(float(arr.min()), 4),
                    "max":      round(float(arr.max()), 4),
                    "mean":     round(float(arr.mean()), 4),
                    "std":      round(float(arr.std()), 4),
                    "skewness": round(float(scipy_stats.skew(arr)), 4),
                }
        elif dtype == "text":
            lengths = s.dropna().astype(str).str.len()
            info["avg_len"] = round(float(lengths.mean()), 1) if not lengths.empty else 0
            info["max_len"] = int(lengths.max()) if not lengths.empty else 0
        cols.append(info)
    return cols


# ── Metric value extractor ────────────────────────────────────────────────────

def _metric_val(output: str) -> Optional[float]:
    for line in (output or "").splitlines():
        if line.startswith("METRIC_VALUE:"):
            try:
                return float(line.replace("METRIC_VALUE:", "").strip())
            except ValueError:
                pass
    return None


# ── Aggregation ───────────────────────────────────────────────────────────────

_CATEGORY_MAP = {
    "universal_validity":   "validity",
    "universal_fidelity":   "fidelity",
    "universal_diversity":  "diversity",
    "readability":          "readability",
}

_GRADE_THRESHOLDS = [
    (0.85, "A"),
    (0.70, "B"),
    (0.55, "C"),
    (0.40, "D"),
]


def _grade(score: float) -> str:
    for threshold, letter in _GRADE_THRESHOLDS:
        if score >= threshold:
            return letter
    return "F"


def aggregate(results: List[Dict]) -> Dict:
    computed = [r for r in results if not r.get("error")]
    skipped  = [r for r in results if r.get("error")]

    if not computed:
        return {
            "overall_score":   None,
            "grade":           "N/A",
            "by_category":     {},
            "metric_count":    len(results),
            "computed_count":  0,
            "skipped_count":   len(skipped),
            "anomaly_summary": {},
        }

    scores  = [r["quality_score"] for r in computed]
    overall = float(np.mean(scores))

    # Per-category breakdown
    buckets: Dict[str, List[float]] = {}
    for r in computed:
        raw_cat = r.get("metric_type", "other")
        cat     = _CATEGORY_MAP.get(raw_cat, raw_cat.replace("universal_", ""))
        buckets.setdefault(cat, []).append(r["quality_score"])

    by_category = {}
    for cat, cat_scores in sorted(buckets.items()):
        by_category[cat] = {
            "mean_score": round(float(np.mean(cat_scores)), 4),
            "min_score":  round(float(np.min(cat_scores)), 4),
            "max_score":  round(float(np.max(cat_scores)), 4),
            "count":      len(cat_scores),
            "metrics":    [],
        }

    for r in computed:
        raw_cat = r.get("metric_type", "other")
        cat     = _CATEGORY_MAP.get(raw_cat, raw_cat.replace("universal_", ""))
        if cat in by_category:
            by_category[cat]["metrics"].append({
                "name":      r["metric_name"],
                "score":     round(r["quality_score"], 4),
                "anomalies": r.get("anomalous_row_count", 0),
            })

    anomaly_summary = {
        r["metric_name"]: r["anomalous_row_count"]
        for r in computed
        if r.get("anomalous_row_count", 0) > 0
    }

    # Top anomalous rows across all metrics (union, sorted by frequency)
    from collections import Counter
    row_counts: Counter = Counter()
    for r in computed:
        for idx in r.get("anomalous_row_indices", []):
            row_counts[idx] += 1
    most_problematic = [
        {"row": idx, "flagged_by": cnt}
        for idx, cnt in row_counts.most_common(20)
    ]

    return {
        "overall_score":       round(overall, 4),
        "grade":               _grade(overall),
        "by_category":         by_category,
        "metric_count":        len(results),
        "computed_count":      len(computed),
        "skipped_count":       len(skipped),
        "anomaly_summary":     anomaly_summary,
        "most_problematic_rows": most_problematic,
    }


# ── Registry tool runner ──────────────────────────────────────────────────────

_REGISTRY_TOOLS = [
    "flesch_reading_ease",
    "lexical_density",
    "stopword_ratio",
    "average_sentence_length",
    "perplexity",
    "mauve",
]


def _run_registry_tools(
    registry: ToolRegistry, df: pd.DataFrame, primary_col: str
) -> List[Dict]:
    results = []
    for tool_name in _REGISTRY_TOOLS:
        fn = registry.get(tool_name)
        if fn is None:
            print(f"[BENCH]   {tool_name:<48}  SKIP  (not available)")
            results.append({
                "metric_name":         tool_name,
                "metric_type":         "readability",
                "metric_category":     "readability",
                "metric_source":       "tool_registry",
                "target_column":       primary_col,
                "quality_score":       0.0,
                "error":               "tool not available in this environment",
                "anomalous_row_count": 0,
                "anomalous_row_indices": [],
            })
            continue

        try:
            value, anomalous, output = fn(df, primary_col)
            quality = float(np.clip(value, 0.0, 1.0))
            raw_val = _metric_val(output)
            interp  = next(
                (l for l in output.splitlines() if l and not l.startswith("METRIC_VALUE:")),
                output[:100],
            )
            print(f"[BENCH]   {tool_name:<48}  {quality:.4f}")
            results.append({
                "metric_name":           tool_name,
                "metric_type":           "readability",
                "metric_category":       "readability",
                "metric_source":         "tool_registry",
                "target_column":         primary_col,
                "quality_score":         quality,
                "raw_value":             raw_val,
                "execution_output":      output,
                "error":                 None,
                "anomalous_row_count":   len(anomalous),
                "anomalous_row_indices": anomalous[:20],
                "metric_interpretation": interp,
            })
        except Exception as exc:
            print(f"[BENCH]   {tool_name:<48}  ERROR ({exc})")
            results.append({
                "metric_name":         tool_name,
                "metric_type":         "readability",
                "metric_category":     "readability",
                "metric_source":       "tool_registry",
                "target_column":       primary_col,
                "quality_score":       0.0,
                "error":               str(exc),
                "anomalous_row_count": 0,
                "anomalous_row_indices": [],
            })
    return results


# ── Main entry point ──────────────────────────────────────────────────────────

def run_benchmark(
    dataset,
    dataset_name: str = "dataset",
    max_rows: int = MAX_ROWS_FOR_EVAL,
) -> Dict:
    """
    Run the full benchmarking pipeline on a HuggingFace Dataset object.
    Returns a complete benchmark report dict.
    """
    print(f"\n{'═' * 65}")
    print(f"  BENCHMARK  ·  {dataset_name}")
    print(f"{'═' * 65}\n")

    # ── 1. Load & cap ────────────────────────────────────────────────
    df = dataset.to_pandas()
    try:
        df = df.convert_dtypes(dtype_backend="numpy_nullable")
    except Exception:
        pass

    original_rows = len(df)
    if original_rows > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
        print(f"[BENCH] Capped {original_rows:,} → {max_rows:,} rows")

    print(f"[BENCH] Shape  : {len(df):,} rows × {len(df.columns)} cols")

    # ── 2. Column metadata ───────────────────────────────────────────
    col_stats = _extract_col_stats(df)
    try:
        dup_count = int(df.duplicated().sum())
    except Exception:
        try:
            dup_count = int(df.astype(str).duplicated().sum())
        except Exception:
            dup_count = 0

    metadata = {
        "row_count":      len(df),
        "col_count":      len(df.columns),
        "original_rows":  original_rows,
        "duplicate_rows": dup_count,
        "duplicate_pct":  round(dup_count / max(len(df), 1) * 100, 2),
        "null_pct_mean":  round(float(df.isnull().mean().mean() * 100), 2),
        "columns":        col_stats,
    }

    print(f"[BENCH] Columns: {len(df.columns)}  |  "
          f"Duplicates: {dup_count} ({metadata['duplicate_pct']:.1f}%)  |  "
          f"Nulls: {metadata['null_pct_mean']:.1f}%")

    # ── 3. Text column detection ─────────────────────────────────────
    text_cols = _detect_text_cols(df)
    if not text_cols:
        print("[BENCH] No text columns found — cannot compute text quality metrics.")
        return {
            "dataset_name": dataset_name,
            "timestamp":    datetime.datetime.utcnow().isoformat(),
            "metadata":     metadata,
            "text_columns": [],
            "metrics":      [],
            "summary": {
                "overall_score": None,
                "grade":         "N/A",
                "message":       "No text columns detected.",
            },
        }

    primary_col = max(
        text_cols,
        key=lambda c: df[c].dropna().astype(str).str.len().mean(),
    )
    print(f"[BENCH] Text cols : {text_cols}")
    print(f"[BENCH] Primary   : '{primary_col}'")

    all_results: List[Dict] = []

    # ── 4. Universal Text Metrics ────────────────────────────────────
    print(f"\n[BENCH] ── UNIVERSAL TEXT METRICS ──────────────────────────────")
    utm = UniversalTextMetrics()
    universal = utm.compute_all(df, text_cols, thinker_output={})
    all_results.extend(universal)

    # ── 5. Tool Registry metrics ─────────────────────────────────────
    print(f"\n[BENCH] ── TOOL REGISTRY METRICS ───────────────────────────────")
    registry = ToolRegistry()
    registry_results = _run_registry_tools(registry, df, primary_col)
    all_results.extend(registry_results)

    # ── 6. Aggregate ─────────────────────────────────────────────────
    print(f"\n[BENCH] ── AGGREGATING ─────────────────────────────────────────")
    summary = aggregate(all_results)

    print(f"[BENCH] Overall Score : {summary['overall_score']:.4f}   "
          f"Grade: {summary['grade']}")
    print(f"[BENCH] Computed: {summary['computed_count']}  "
          f"Skipped: {summary['skipped_count']}")
    for cat, info in summary["by_category"].items():
        print(f"[BENCH]   {cat:<20}  avg={info['mean_score']:.4f}  "
              f"({info['count']} metrics)")

    return {
        "dataset_name":   dataset_name,
        "timestamp":      datetime.datetime.utcnow().isoformat(),
        "metadata":       metadata,
        "text_columns":   text_cols,
        "primary_column": primary_col,
        "metrics":        all_results,
        "summary":        summary,
    }
