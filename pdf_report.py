"""
pdf_report.py — PDF generation for SynAgent audit reports.
Importable standalone (no Streamlit dependency).
"""

from datetime import datetime


def _extract_metric_value(output: str) -> str:
    for line in (output or "").splitlines():
        if line.startswith("METRIC_VALUE:"):
            return line.replace("METRIC_VALUE:", "").strip()
    return ""


def build_pdf(state: dict, mode: str = "full") -> bytes:
    """
    Build a PDF from pipeline state dict.
      mode="full"    — Verdict + Universal + Research + Thinker + Researcher
      mode="summary" — Verdict + Universal Metrics + Research Metrics only
    """
    from fpdf import FPDF

    verdict_map = {
        "HIGH_QUALITY":       "HIGH QUALITY",
        "ACCEPTABLE_QUALITY": "ACCEPTABLE QUALITY",
        "POOR_QUALITY":       "POOR QUALITY",
        "UNKNOWN":            "UNKNOWN",
    }
    verdict_color = {
        "HIGH_QUALITY":       (34, 197, 94),
        "ACCEPTABLE_QUALITY": (234, 179, 8),
        "POOR_QUALITY":       (239, 68, 68),
        "UNKNOWN":            (148, 163, 184),
    }

    universal  = state.get("universal_metric_results",  []) or []
    per_metric = state.get("per_metric_results",        []) or []
    thinker    = state.get("thinker_output",            {}) or {}
    researcher = state.get("researcher_output",         {}) or {}
    evaluator  = state.get("evaluator_output",          {}) or {}

    verdict     = evaluator.get("final_verdict", "UNKNOWN")
    vcolor      = verdict_color.get(verdict, (148, 163, 184))
    verdict_str = verdict_map.get(verdict, verdict)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    # ── Header ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "SynAgent - Data Quality Audit Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    label = "Full Report" if mode == "full" else "Summary Report (Verdict + Metrics)"
    pdf.cell(0, 6, f"{label}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             ln=True, align="C")
    pdf.ln(6)

    # ── Helpers ───────────────────────────────────────────────────────
    def _safe(text: str) -> str:
        """Convert any input to a safe PDF string."""
        # Handle bytes/bytearray input
        if isinstance(text, (bytes, bytearray)):
            try:
                t = text.decode('utf-8', errors='replace')
            except:
                t = str(text)
        else:
            t = str(text)
        
        # Replace problematic Unicode characters
        t = t.replace("—", "-").replace("–", "-").replace("'", "'") \
             .replace(""", '"').replace(""", '"').replace("•", "*") \
             .replace("→", "->").replace("←", "<-").replace("·", ".")
        
        # Ensure it's latin-1 safe
        try:
            return t.encode("latin-1", "replace").decode("latin-1")
        except:
            return t

    def _section(title: str):
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"  {_safe(title)}", ln=True, fill=True)
        pdf.set_text_color(15, 23, 42)
        pdf.ln(2)

    def _mc(text: str, h: int = 6):
        """multi_cell that always starts from the left margin."""
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, h, _safe(text))

    def _body(text: str, size: int = 10):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(30, 41, 59)
        _mc(text, 6)
        pdf.ln(1)

    def _kv(label: str, value: str):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 5, f"{_safe(label)}:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 41, 59)
        _mc(_safe(str(value))[:800], 5)
        pdf.ln(1)

    def _metric_row(name: str, value: str, anomalies: int, quality: float, error: str = ""):
        pdf.set_font("Helvetica", "", 8)
        if error:
            pdf.set_text_color(150, 150, 150)
        elif quality >= 0.75:
            pdf.set_text_color(22, 163, 74)
        elif quality >= 0.50:
            pdf.set_text_color(202, 138, 4)
        else:
            pdf.set_text_color(220, 38, 38)
        # Even smaller widths that fit: 50 + 10 + 8 + remaining
        pdf.cell(50, 5, _safe(name)[:35], ln=False)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(10, 5, _safe(value)[:8] if value else "-", ln=False)
        pdf.cell(8, 5, str(int(anomalies))[:2], ln=False)
        pdf.cell(0,  5, f"{quality:.2f}" if not error else "E", ln=True)

    def _metrics_table_header():
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(50, 5, "Metric",    ln=False)
        pdf.cell(10, 5, "Val",       ln=False)
        pdf.cell(8, 5, "An",         ln=False)
        pdf.cell(0,  5, "Qual",      ln=True)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(1)

    # ── 1. Verdict banner ─────────────────────────────────────────────
    _section("VERDICT")
    pdf.set_fill_color(*vcolor)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, f"  {verdict_str}", ln=True, fill=True)
    pdf.set_text_color(15, 23, 42)
    pdf.ln(2)

    if evaluator.get("verdict_reasoning"):
        _body(evaluator["verdict_reasoning"])
    if evaluator.get("dataset_level_evidence"):
        _kv("Evidence",   evaluator["dataset_level_evidence"])
    if evaluator.get("quality_observations"):
        _kv("Observations", evaluator["quality_observations"])
    if evaluator.get("statistical_justification"):
        _kv("Statistics", evaluator["statistical_justification"])

    by_metric = evaluator.get("quality_by_metric", [])
    if by_metric:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 7, "  Per-Metric Quality Assessment", ln=True, fill=True)
        level_colors = {
            "GOOD":       (22, 163, 74),
            "ACCEPTABLE": (202, 138, 4),
            "POOR":       (220, 38, 38),
            "HIGH":       (220, 38, 38),
        }
        for m in by_metric:
            lvl = m.get("quality_level", m.get("risk_level", ""))
            col = level_colors.get(lvl, (100, 116, 139))
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*col)
            mname = _safe(str(m.get("metric", "")))
            pdf.cell(0, 5, f"  [{lvl}] {mname[:60]}", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(30, 41, 59)
            _mc("    " + _safe(str(m.get("finding", "")))[:120], 5)
            pdf.ln(1)
    pdf.ln(4)

    # ── 2. Universal Metrics ──────────────────────────────────────────
    _section("UNIVERSAL TEXT METRICS")
    _metrics_table_header()
    for r in universal:
        val = _extract_metric_value(r.get("execution_output", ""))
        _metric_row(
            r.get("metric_name", ""),
            val,
            r.get("anomalous_row_count", 0),
            r.get("quality_score", 0.0),
            r.get("error", "") or "",
        )
    pdf.ln(4)

    # ── 3. Research Metrics ───────────────────────────────────────────
    research_only = [r for r in per_metric if r.get("metric_source") != "universal"]
    if research_only:
        _section("RESEARCH METRICS")
        _metrics_table_header()
        for r in research_only:
            val = _extract_metric_value(r.get("execution_output", ""))
            _metric_row(
                r.get("metric_name", ""),
                val,
                r.get("anomalous_row_count", 0),
                r.get("quality_score", 0.0),
                r.get("error", "") or "",
            )

        for r in research_only:
            interp = r.get("metric_interpretation", "")
            if interp:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 5, f"  {_safe(r.get('metric_name',''))}:", ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(30, 41, 59)
                _mc(_safe(interp[:500]), 5)
        pdf.ln(4)

    # ── 4. Thinker (full mode only) ───────────────────────────────────
    if mode == "full" and thinker:
        pdf.add_page()
        _section("THINKER - DATASET UNDERSTANDING")
        dom = thinker.get("domain", {})
        _kv("Dataset Type", thinker.get("dataset_type", "-"))
        _kv("Domain",       f"{dom.get('name','-')} ({dom.get('confidence',0):.0%} confidence)")
        _kv("Reasoning",    dom.get("reasoning", ""))
        pdf.ln(2)

        hints = thinker.get("surface_quality_hints", [])
        if hints:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 6, "Surface Quality Hints:", ln=True)
            for h in hints:
                _body(f"  - {h}", size=9)

        sec = thinker.get("governance_warnings", [])
        if sec:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 6, "Security Recommendations:", ln=True)
            for w in sec:
                _body(f"  ! {w}", size=9)

        col_profiles = thinker.get("column_profiles", [])
        if col_profiles:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 6, "Column Profiles:", ln=True)
            for cp in col_profiles:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 5, _safe(f"  {cp.get('column_name','')} - {cp.get('inferred_dtype','')}, {cp.get('semantic_role','')}"), ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(30, 41, 59)
                if cp.get("notes"):
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(100, 116, 139)
                    _mc(_safe("    " + cp["notes"][:200]), 5)
        pdf.ln(4)

    # ── 5. Researcher (full mode only) ────────────────────────────────
    if mode == "full" and researcher:
        _section("RESEARCHER - PROPOSED METRICS & SOURCES")
        metrics_list = researcher.get("final_metrics") or researcher.get("proposed_metrics", [])
        summary = researcher.get("research_summary", "")
        if summary:
            _body(summary)

        for i, m in enumerate(metrics_list, 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 41, 59)
            mname = f"{i}. {m.get('metric_name','?')} [{m.get('metric_type','')}]"
            pdf.cell(0, 6, _safe(mname), ln=True)
            if m.get("description"):
                _body(f"   {m['description']}", size=9)
            for c in m.get("paper_citations", []):
                if c.get("title"):
                    cite = f"   > {c.get('title','')[:80]} [{c.get('arxiv_id','')}]"
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(100, 116, 139)
                    _mc(_safe(cite), 5)
            pdf.ln(1)

    # Ensure output is bytes
    try:
        pdf_output = pdf.output()
        # Handle both str and bytes return types
        if isinstance(pdf_output, bytes):
            return pdf_output
        elif isinstance(pdf_output, bytearray):
            return bytes(pdf_output)
        else:
            # It's a string, encode it
            return pdf_output.encode('utf-8')
    except Exception as e:
        # Fallback: try without encoding
        return pdf.output()  # Returns bytes by default in newer fpdf2
