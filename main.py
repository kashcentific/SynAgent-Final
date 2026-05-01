# main.py

import argparse
import io
import json
import os
import re
import sys

from graph import build_graph
from config import REPORT_OUTPUT_PATH
from visualize import generate_pipeline_graph

BENCHMARK_REPORT_PATH   = "benchmarking_results.json"
BENCHMARK_WORKFLOW_PATH = "benchmarking_workflow.log"


# ──────────────────────────────────────────────────────────────────────
# Tee logger — mirrors everything printed to console into a log file
# ──────────────────────────────────────────────────────────────────────

class _Tee:
    """Writes to both the real stdout and a log file simultaneously."""
    def __init__(self, stream, log_path: str):
        self._stream  = stream
        self._logfile = open(log_path, "w", encoding="utf-8", errors="replace")

    def write(self, data):
        self._stream.write(data)
        self._logfile.write(data)

    def flush(self):
        self._stream.flush()
        self._logfile.flush()

    def close(self):
        self._logfile.close()

    # Delegate everything else (isatty, fileno, etc.) to the real stream
    def __getattr__(self, name):
        return getattr(self._stream, name)


# ──────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────

def _bar(char: str = "─", width: int = 70) -> str:
    return char * width


def _section(title: str, char: str = "─"):
    print(f"\n{_bar(char)}")
    print(f"  {title}")
    print(_bar(char))


def _sub(title: str):
    print(f"\n  ┌─ {title}")


def _row(label: str, value: str, indent: int = 4):
    pad = " " * indent
    # Wrap long values
    if len(value) > 100:
        print(f"{pad}{label}:")
        for line in value.splitlines():
            print(f"{pad}  {line}")
    else:
        print(f"{pad}{label}: {value}")


# ──────────────────────────────────────────────────────────────────────
# Display: thinker
# ──────────────────────────────────────────────────────────────────────

def _display_thinker(out: dict):
    if not out:
        print("\n[!] No thinker output.")
        return

    _section("② THINKER — DATASET UNDERSTANDING", "═")

    domain = out.get("domain", {})
    print(f"\n  Dataset Type : {out.get('dataset_type', '?')}")
    print(
        f"  Domain       : {domain.get('name', '?')} "
        f"(confidence {domain.get('confidence', 0):.0%})"
    )
    print(f"  Domain Why   : {domain.get('reasoning', '')}")

    _sub("Column Profiles")
    for cp in out.get("column_profiles", []):
        print(
            f"  │  {cp.get('column_name', ''):<25} "
            f"dtype={cp.get('inferred_dtype', ''):<12} "
            f"role={cp.get('semantic_role', '')}"
        )
        if cp.get("notes"):
            print(f"  │    ↳ {cp['notes']}")

    hints = out.get("surface_quality_hints", [])
    if hints:
        _sub("Surface Quality Hints")
        for h in hints:
            print(f"  │  • {h}")

    warnings = out.get("governance_warnings", [])
    if warnings:
        _sub("Governance Warnings")
        for w in warnings:
            print(f"  │  ⚠  {w}")

    ref = out.get("reference_dataset", {})
    _sub("Reference Dataset Assessment")
    print(f"  │  Needed : {ref.get('seems_needed', '?')}")
    print(f"  │  Why    : {ref.get('reasoning', '')}")

    _sub("Recommended Downstream Agents")
    for name, info in out.get("recommended_agents", {}).items():
        flag = "✓ RUN " if info.get("run") else "✗ SKIP"
        print(f"  │  [{flag}] {name:<12} — {info.get('reason', '')}")

    notes = out.get("execution_notes", [])
    if notes:
        _sub("Execution Notes for Downstream Agents")
        for n in notes:
            print(f"  │  → {n}")


# ──────────────────────────────────────────────────────────────────────
# Display: researcher
# ──────────────────────────────────────────────────────────────────────

def _display_researcher(out: dict):
    if not out:
        print("\n[!] No researcher output.")
        return

    _section("③ RESEARCHER — EXTERNAL RESEARCH & METRIC PROPOSALS", "═")

    # Show what each tool actually returned
    context = out.get("research_context", {})
    if context:
        _sub("Research Tool Results")

        # Multi-angle DuckDuckGo
        ddg = context.get("duckduckgo", {})
        if ddg:
            print(f"  │")
            print(f"  │  🌐 DuckDuckGo (multi-angle)")
            for i, q in enumerate(ddg.get("queries", []), 1):
                print(f"  │    Angle {i}: {q}")
            print(f"  │    Unique URLs : {ddg.get('unique_urls', '?')}   Scraped pages: {ddg.get('scraped_pages', '?')}")

        # Wikipedia
        wiki = context.get("wikipedia", {})
        if wiki:
            print(f"  │")
            print(f"  │  📖 Wikipedia")
            print(f"  │    Query : \"{wiki.get('query', '')}\"")
            excerpt = wiki.get("excerpt", "").replace("\n", " ").strip()
            if excerpt:
                print(f"  │    Found : {excerpt[:200]}")

        # ArXiv papers — shown whether PDF loaded or abstract-only
        print(f"  │")
        arxiv_papers = context.get("arxiv_papers", [])
        pdf_ok   = sum(1 for p in arxiv_papers if p.get("pdf_loaded"))
        abs_only = len(arxiv_papers) - pdf_ok
        print(f"  │  [ArXiv] Deep Research — {len(arxiv_papers)} papers "
              f"({pdf_ok} full PDF, {abs_only} abstract-only)")
        if not arxiv_papers:
            print(f"  │    (no papers returned — check ArXiv query or network)")
        for i, p in enumerate(arxiv_papers, 1):
            loaded = "PDF  " if p.get("pdf_loaded") else "abst."
            chunks = p.get("excerpts_count", 0)
            print(f"  │    [{i}][{loaded}] \"{p.get('title', '?')[:60]}\"")
            print(f"  │           ID: {p.get('arxiv_id', '?')}  Year: {p.get('year', '?')}  "
                  f"Relevant chunks: {chunks}")

    # Research summary
    summary = out.get("research_summary", "")
    if summary:
        _sub("Research Summary")
        print(f"  │  {summary}")

    # Proposed metrics with source traceability
    # Final metrics (populated after consolidation)
    metrics = out.get("final_metrics", out.get("proposed_metrics", []))
    if metrics:
        def _cite_count(m):
            c = m.get("paper_citations", [])
            if not isinstance(c, list):
                c = [m.get("paper_citation", {})] if m.get("paper_citation") else []
            return len([x for x in c if isinstance(x, dict) and (x.get("arxiv_id") or x.get("title"))])
        cited = sum(1 for m in metrics if _cite_count(m) >= 2)
        _sub(f"Metrics ({len(metrics)} total, {cited} with ≥2 paper citations)")
        for i, m in enumerate(metrics, 1):
            # Normalise: support both paper_citations list and legacy paper_citation dict
            cites = m.get("paper_citations", [])
            if not cites:
                single = m.get("paper_citation", {})
                if single.get("arxiv_id") or single.get("title"):
                    cites = [single]

            cite_ids = ", ".join(c.get("arxiv_id", "") for c in cites if c.get("arxiv_id"))
            cite_tag = f" 📄 [{cite_ids}]" if cite_ids else ""
            print(f"\n  │  {i:>2}. {m.get('metric_name')} [{m.get('metric_type')}]{cite_tag}")
            print(f"  │      Description    : {m.get('description', '')}")
            print(f"  │      Reasoning      : {m.get('reasoning', '')}")
            src_inf = m.get("source_influence", "")
            if src_inf:
                print(f"  │      Source         : {src_inf}")
            print(f"  │      Execution Hint : {m.get('execution_hint', '')}")
            for ci, cite in enumerate(cites, 1):
                if cite.get("supporting_text") or cite.get("title"):
                    print(f"  │      Paper {ci}        : \"{cite.get('title', '')}\" [{cite.get('arxiv_id', '')}]")
                    if cite.get("supporting_text"):
                        print(f"  │        Quote        : \"{cite['supporting_text'][:160]}\"")
                        print(f"  │                       — {cite.get('authors', '')} ({cite.get('year', '')})")


# ──────────────────────────────────────────────────────────────────────
# Display: universal text metrics (dedicated section)
# ──────────────────────────────────────────────────────────────────────

def _display_universal_metrics(universal: list):
    if not universal:
        return

    _section("① UNIVERSAL TEXT METRICS — Validity · Fidelity · Diversity · Safety", "═")

    categories = ["validity", "fidelity", "diversity", "safety"]
    cat_icons  = {"validity": "V", "fidelity": "F", "diversity": "D", "safety": "S"}

    for cat in categories:
        group = [r for r in universal if r.get("metric_category") == cat]
        if not group:
            continue
        print(f"\n  [{cat_icons.get(cat, cat[0].upper())}] {cat.upper()}")
        for r in group:
            err = r.get("error")
            val_line = ""
            for line in (r.get("execution_output") or "").splitlines():
                if line.startswith("METRIC_VALUE:"):
                    val_line = line.replace("METRIC_VALUE:", "").strip()
                    break
            if err:
                status = f"SKIPPED — {err}"
            elif val_line:
                status = f"score = {val_line}"
            else:
                status = "OK"
            print(f"  │  {r['metric_name']:<48} {status}")
            if not err and r.get("anomalous_row_count", 0) > 0:
                print(f"  │    -> {r['anomalous_row_count']} anomalous rows flagged")


# ──────────────────────────────────────────────────────────────────────
# Display: evaluator (research metrics only)
# ──────────────────────────────────────────────────────────────────────

def _display_evaluator(evaluator_out: dict, per_metric: list):
    if not evaluator_out and not per_metric:
        print("\n[!] No evaluator output.")
        return

    # Research metrics displayed separately from universal
    research_metrics = [r for r in per_metric if r.get("metric_source") != "universal"]

    _section("④ RESEARCH METRICS — COMPUTATION & FINDINGS", "═")

    for idx, r in enumerate(research_metrics, 1):
        name   = r.get("metric_name", "?")
        err    = r.get("error")
        acount = r.get("anomalous_row_count", 0)
        status = "✗ ERROR" if err else "✓ COMPUTED"

        print(f"\n  ┌─ [{idx}] {name}  [{status}]")

        # Why this metric was selected
        sel_reason = r.get("selection_reason", "")
        if sel_reason:
            print(f"  │  Why selected : {sel_reason}")

        # Which research source drove this metric
        src_inf = r.get("source_influence", "")
        if src_inf:
            print(f"  │  Research source : {src_inf}")

        # Execution output (the actual numbers)
        exec_out = (r.get("execution_output") or "").strip()
        if exec_out:
            print(f"  │  Execution output:")
            for line in exec_out.splitlines():
                print(f"  │    {line}")

        if err:
            print(f"  │  ⚠  Error : {err}")

        # Row-level findings
        print(f"  │  Anomalous rows : {acount}")
        narratives = r.get("anomalous_row_narratives", [])
        if narratives:
            print(f"  │  Flagged rows:")
            for n in narratives[:8]:
                print(f"  │    ⚠  {n}")

        # LLM interpretation
        interp = r.get("metric_interpretation", "")
        if interp:
            print(f"  │  Interpretation:")
            for line in interp.strip().splitlines():
                print(f"  │    {line}")

        print(f"  └{'─' * 66}")

    # Final verdict
    if evaluator_out:
        _section("⑤ FINAL VERDICT", "═")

        verdict = evaluator_out.get("final_verdict", "UNKNOWN")
        emoji   = {
            "HIGH_QUALITY"       : "✅",
            "ACCEPTABLE_QUALITY" : "⚠️ ",
            "POOR_QUALITY"       : "❌",
            # legacy fallbacks
            "SAFE"               : "✅",
            "USABLE_WITH_CAUTION": "⚠️ ",
            "UNSAFE"             : "❌",
        }.get(verdict, "❓")

        print(f"\n  {emoji}  DATA QUALITY VERDICT: {verdict}")
        print(f"\n  Reasoning:")
        for line in evaluator_out.get("verdict_reasoning", "").splitlines():
            print(f"    {line}")

        print(f"\n  Dataset-level Evidence:")
        for line in evaluator_out.get("dataset_level_evidence", "").splitlines():
            print(f"    {line}")

        obs = evaluator_out.get("quality_observations", evaluator_out.get("sample_level_inconsistencies", ""))
        if obs:
            print(f"\n  Quality Observations:")
            for line in obs.splitlines():
                print(f"    {line}")

        print(f"\n  Statistical Justification:")
        for line in evaluator_out.get("statistical_justification", "").splitlines():
            print(f"    {line}")

        quality = evaluator_out.get("quality_by_metric", evaluator_out.get("risk_by_metric", []))
        if quality:
            _sub("Quality by Metric")
            for rb in quality:
                lvl  = rb.get("quality_level", rb.get("risk_level", "?"))
                icon = {"GOOD": "🟢", "ACCEPTABLE": "🟡", "POOR": "🔴",
                        "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(lvl, "⚪")
                print(f"  │  {icon} [{lvl:<10}] {rb.get('metric', '')}")
                print(f"  │             Finding : {rb.get('finding', '')}")
                note = rb.get("note", rb.get("why", ""))
                if note:
                    print(f"  │             Note    : {note}")


# ──────────────────────────────────────────────────────────────────────
# Save JSON report
# ──────────────────────────────────────────────────────────────────────

def _save_report(state: dict):
    report = {
        "universal_metric_results": state.get("universal_metric_results", []),
        "thinker_output"          : state.get("thinker_output", {}),
        "researcher_output"       : state.get("researcher_output", {}),
        "per_metric_results"      : state.get("per_metric_results", []),
        "evaluator_output"        : state.get("evaluator_output", {}),
        "errors"                  : state.get("errors", []),
    }
    # Strip generated code from JSON report (it's verbose; kept on console)
    for r in report.get("per_metric_results", []):
        r.pop("generated_code", None)

    out_path = BENCHMARK_REPORT_PATH
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  [OK] Structured report saved  → {os.path.abspath(out_path)}")
    except Exception as exc:
        print(f"\n  [WARN] Could not save report: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Master display
# ──────────────────────────────────────────────────────────────────────

def display_result(state: dict):
    # Universal metrics always shown first — they're computed before any LLM agent
    _display_universal_metrics(state.get("universal_metric_results", []))
    _display_thinker(state.get("thinker_output", {}))
    _display_researcher(state.get("researcher_output", {}))
    _display_evaluator(
        state.get("evaluator_output", {}),
        state.get("per_metric_results", []),
    )

    errs = state.get("errors", [])
    if errs:
        _section("PIPELINE ERRORS", "!")
        for e in errs:
            print(f"  [ERR] {e}")

    _section("REPORT", "═")
    _save_report(state)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


def _parse_hf_name(raw: str) -> str:
    m = re.search(r'load_dataset\s*\(\s*["\']([^"\']+)["\']', raw)
    if m:
        return m.group(1).strip()
    m = re.search(r'huggingface\.co/datasets/([^\s\'"]+)', raw)
    if m:
        return m.group(1).strip().rstrip("/")
    return raw.strip()


def main():
    # ── Force UTF-8 stdout on Windows before anything else ───────────
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

    parser = argparse.ArgumentParser(
        description="SynAgent - Data Quality Audit pipeline"
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--csv",     metavar="PATH",    help="Path to a CSV file")
    src.add_argument("--dataset", metavar="HF_NAME", help="HuggingFace dataset id or load_dataset() snippet")
    parser.add_argument("--rows", type=int, default=1000,
                        help="Number of rows to audit (default: 1000)")
    parser.add_argument("--split", default=None,
                        help="HuggingFace split string, e.g. train[:500] (default: train[:<rows>])")
    parser.add_argument("--hint", default=None,
                        help="Optional natural-language hint about the dataset for the agents")
    args = parser.parse_args()

    # ── Start Tee logger before anything is printed ──────────────────
    tee = _Tee(sys.stdout, BENCHMARK_WORKFLOW_PATH)
    sys.stdout = tee

    try:
        _run_pipeline(args)
    finally:
        sys.stdout = tee._stream
        tee.close()


def _run_pipeline(args):
    import pandas as pd
    from datasets import Dataset, load_dataset as _load_hf

    # ── Load dataset ─────────────────────────────────────────────────
    if args.csv:
        dataset_name = os.path.splitext(os.path.basename(args.csv))[0]
        _section(f"SYNAGENT — DATA QUALITY AUDIT  [{dataset_name}]", "═")
        print(f"\n  Report   -> {BENCHMARK_REPORT_PATH}")
        print(f"  Workflow -> {BENCHMARK_WORKFLOW_PATH}")
        print(f"\n  Loading CSV: {args.csv} ...")
        try:
            df = pd.read_csv(args.csv)
            if len(df) > args.rows:
                df = df.head(args.rows).reset_index(drop=True)
            ds = Dataset.from_pandas(df, preserve_index=False)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    elif args.dataset:
        dataset_name = _parse_hf_name(args.dataset).split("/")[-1]
        _section(f"SYNAGENT — DATA QUALITY AUDIT  [{dataset_name}]", "═")
        print(f"\n  Report   -> {BENCHMARK_REPORT_PATH}")
        print(f"  Workflow -> {BENCHMARK_WORKFLOW_PATH}")
        split = args.split or f"train[:{args.rows}]"
        hf_id = _parse_hf_name(args.dataset)
        print(f"\n  Loading HuggingFace dataset: {hf_id}  (split={split}) ...")
        try:
            raw_ds = _load_hf(hf_id, split=split)
            df = raw_ds.to_pandas()
            ds = Dataset.from_pandas(df, preserve_index=False)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    else:
        # default fallback — original hardcoded dataset
        dataset_name = "SWE-agent-trajectories"
        _section(f"SYNAGENT — DATA QUALITY AUDIT  [{dataset_name}]", "═")
        print(f"\n  Report   -> {BENCHMARK_REPORT_PATH}")
        print(f"  Workflow -> {BENCHMARK_WORKFLOW_PATH}")
        split = args.split or f"train[:{args.rows}]"
        print(f"\n  Loading dataset: nebius/SWE-agent-trajectories  (split={split}) ...")
        try:
            raw_ds = _load_hf("nebius/SWE-agent-trajectories", split=split)
            df = raw_ds.to_pandas()
            ds = Dataset.from_pandas(df, preserve_index=False)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    print(f"  [OK] {len(ds)} rows x {len(ds.column_names)} columns")
    print(f"  Columns: {ds.column_names}")
    for col in ds.column_names:
        s = df[col].dropna().astype(str)
        avg_len = s.str.len().mean()
        print(f"    {col:<30}  avg_len={avg_len:.0f}  nulls={df[col].isnull().sum()}")

    # ── Build user hint ───────────────────────────────────────────────
    if args.hint:
        user_hint = args.hint
    else:
        col_desc = ", ".join(ds.column_names)
        sample_vals = {}
        for col in ds.column_names[:3]:
            v = str(df[col].dropna().iloc[0])[:120] if len(df[col].dropna()) else ""
            sample_vals[col] = v
        samples_str = "  |  ".join(f'{k}: "{v}"' for k, v in sample_vals.items())
        user_hint = (
            f"Dataset: {dataset_name}. "
            f"Columns: {col_desc}. "
            f"Sample values — {samples_str}. "
            f"Evaluate text quality, semantic coherence, formatting consistency, "
            f"and any domain-specific issues relevant to this dataset type."
        )

    # ── Run pipeline ──────────────────────────────────────────────────
    app = build_graph()
    initial_state = {
        "dataset"  : ds,
        "user_hint": user_hint,
        "errors"   : [],
    }

    print("\n  Starting pipeline...\n")
    final_state = app.invoke(initial_state)

    display_result(final_state)

    graph_path = os.path.splitext(BENCHMARK_REPORT_PATH)[0] + "_pipeline_graph.png"
    try:
        generate_pipeline_graph(final_state, output_path=graph_path)
    except Exception as exc:
        print(f"\n  [WARN] Could not generate pipeline graph: {exc}")

    print(f"\n  [OK] Workflow log saved -> {os.path.abspath(BENCHMARK_WORKFLOW_PATH)}")


if __name__ == "__main__":
    main()
