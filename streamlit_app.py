"""
SynAgent — Live Streaming Streamlit UI
"""

import io
import json
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
from datasets import Dataset

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="SynAgent — Data Quality Audit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Stdout → queue tee ────────────────────────────────────────────────────────

_SENTINEL = "__PIPELINE_DONE__"


class _QueueWriter:
    """Tees stdout to a SimpleQueue (single-pipeline use only)."""
    def __init__(self, q: queue.SimpleQueue, orig):
        self._q    = q
        self._orig = orig
        self._buf  = ""

    def write(self, text: str):
        if self._orig:
            try:
                self._orig.write(text)
            except Exception:
                pass
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self._q.put(stripped)

    def flush(self):
        stripped = self._buf.strip()
        if stripped:
            self._q.put(stripped)
            self._buf = ""
        if self._orig:
            try:
                self._orig.flush()
            except Exception:
                pass

    def fileno(self):
        try:
            return self._orig.fileno()
        except Exception:
            raise io.UnsupportedOperation("fileno")

    def isatty(self):
        return False


# ── Log helpers ───────────────────────────────────────────────────────────────

def _get_status_line(lines: list) -> str:
    """
    Scans the most recent log lines and returns a single human-friendly
    sentence describing what the pipeline is doing right now.
    """
    import re as _re
    recent = lines[-30:] if lines else []

    for line in reversed(recent):
        # ── Node completions ───────────────────────────────────────────
        if "NODE COMPLETE: EVALUATOR" in line:
            return "✅  Evaluation complete — building final report…"
        if "NODE COMPLETE: RESEARCHER" in line:
            return "✅  Research complete — handing off to the evaluator…"
        if "NODE COMPLETE: THINKER" in line:
            return "✅  Dataset analysis complete — researching quality metrics…"
        if "NODE COMPLETE: UNIVERSAL_METRICS" in line or "NODE COMPLETE: UNIVERSAL" in line:
            return "✅  Universal metrics computed — starting AI analysis…"
        if "NODE COMPLETE: EXTRACT_METADATA" in line:
            return "✅  Metadata extracted — computing text quality baselines…"
        # ── Evaluator phases ───────────────────────────────────────────
        if "[EVALUATOR] Phase 3" in line or "CONSOLIDATE" in line:
            return "📊  Building final evidence-backed quality report…"
        if "[EVALUATOR] Auto-detected" in line or "auto_semantic" in line:
            return "📊  Running automatic semantic consistency check…"
        m = _re.search(r"Metric\s+(\d+)/(\d+)", line)
        if m and "[EVALUATOR]" in line:
            return f"📊  Computing metric {m.group(1)} of {m.group(2)}…"
        if "[EVALUATOR] Phase 2" in line or "ACT [" in line:
            return "📊  Running metric computations on your dataset…"
        if "[EVALUATOR] Phase 1" in line or "feasibility" in line.lower():
            return "📊  Checking which metrics are feasible on your data…"
        if "[EVALUATOR] Initializing" in line or (
                "[EVALUATOR]" in line and line.strip().startswith("[EVALUATOR]   *")):
            return "📊  Setting up evaluation tools (scipy, sklearn, sentence-transformers)…"
        # ── Researcher phases ──────────────────────────────────────────
        if "[RESEARCHER] Phase 4" in line or "Consolidating" in line:
            return "🔬  Consolidating evidence and finalising metric list…"
        if "[RESEARCHER] Phase 3" in line or "Deep-diving" in line:
            return "🔬  Deep-diving into top metrics with targeted research…"
        if "[RESEARCHER] Phase 2" in line or "Evaluating metric relevance" in line:
            return "🔬  Scoring metric relevance to your dataset…"
        if "[RESEARCHER] Phase 1" in line or "Generating diverse" in line:
            return "🔬  Searching the web and loading ArXiv research papers…"
        if "ArXiv" in line and "Downloading" in line:
            return "🔬  Downloading and reading ArXiv PDFs…"
        if "DuckDuckGo" in line and "unique URLs" in line:
            return "🔬  Scraping web results from DuckDuckGo…"
        if "[RESEARCHER] Initializing" in line or (
                "[RESEARCHER]" in line and "   *" in line):
            return "🔬  Initialising research tools (DuckDuckGo, Wikipedia, ArXiv)…"
        # ── Thinker ────────────────────────────────────────────────────
        if "[THINKING] ✓ Done" in line:
            return "💭  Response received — processing…"
        if "[THINKING]" in line and "Reasoning" in line:
            return "💭  AI is reasoning… (this may take a moment)"
        if "[THINKER]" in line and "reasoning" in line.lower():
            return "🧠  Reasoning about dataset domain and quality profile…"
        if "[THINKER]" in line and "Analyzing" in line:
            return "🧠  Analysing your dataset structure and columns…"
        if "[THINKER]" in line:
            return "🧠  Thinker agent is understanding your dataset…"
        # ── Early nodes ────────────────────────────────────────────────
        if "[UNIVERSAL]" in line:
            return "📏  Computing standard text quality baselines…"
        if "[METADATA]" in line:
            return "🗂️  Extracting dataset metadata and column statistics…"
        if "⚙️" in line:
            return "⚙️  Starting pipeline execution…"
        if "🚀" in line:
            return "🚀  Initialising pipeline…"

    return "⏳  Pipeline is warming up…"


def _render_detail_log(lines: list, height: int = 400) -> None:
    """Colour-coded terminal panel for the full detail log."""

    def _colour(line: str) -> str:
        if any(x in line for x in ("✅", "NODE COMPLETE")):
            return "#4ade80"
        if any(x in line for x in ("❌", "PIPELINE FAILED", "ERROR:")):
            return "#f87171"
        if any(x in line for x in ("🚀", "⚙️", "🏁")):
            return "#93c5fd"
        if "[THINKER]" in line:
            return "#c084fc"
        if "[RESEARCHER]" in line:
            return "#67e8f9"
        if "[EVALUATOR]" in line:
            return "#fb923c"
        if any(x in line for x in ("[METADATA]", "[UNIVERSAL]", "[GRAPH]", "[THINKING]")):
            return "#86efac"
        if line.startswith("═"):
            return "#374151"
        return "#d1d5db"

    def _fmt(line: str) -> str:
        if len(line) > 200:
            line = line[:197] + "…"
        return line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = "".join(
        f'<span style="color:{_colour(l)};display:block;line-height:1.55">{_fmt(l)}</span>'
        for l in lines[-600:]
    ) or '<span style="color:#6b7280">Waiting for output…</span>'

    st.markdown(
        f"""<div style="
            background:#0f172a;border:1px solid #1e293b;border-radius:8px;
            padding:14px 18px;font-family:'Courier New',Courier,monospace;
            font-size:12px;height:{height}px;overflow-y:auto;
            white-space:pre-wrap;word-break:break-all;
        ">{rows}</div>""",
        unsafe_allow_html=True,
    )


# ── Pipeline worker (background thread) ──────────────────────────────────────

def _pipeline_worker(dataset, user_hint, log_q: queue.SimpleQueue, result: dict):
    """Single-analysis worker: redirects stdout so all agent print() goes to the queue."""
    orig       = sys.stdout
    sys.stdout = _QueueWriter(log_q, orig)
    try:
        from graph import build_graph

        log_q.put("🚀  Building pipeline graph...")
        app = build_graph()

        init_state = {"dataset": dataset, "user_hint": user_hint or None, "errors": []}
        log_q.put("⚙️  Streaming execution started…")

        merged:     dict = {}
        nodes_done: list = []

        for chunk in app.stream(init_state):
            node_name  = list(chunk.keys())[0]
            node_delta = chunk[node_name]
            merged.update(node_delta)
            nodes_done.append(node_name)

            log_q.put("═" * 60)
            log_q.put(f"✅  NODE COMPLETE: {node_name.upper()}")
            log_q.put("═" * 60)

            result["completed_nodes"] = list(nodes_done)
            result["state"]           = dict(merged)

        result["done"]  = True
        result["error"] = None
        log_q.put(f"🏁  All {len(nodes_done)} nodes finished successfully.")

    except Exception as exc:
        import traceback
        err = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        result["done"]  = True
        result["error"] = err
        log_q.put(f"❌  PIPELINE FAILED: {err}")
    finally:
        sys.stdout = orig
        log_q.put(_SENTINEL)


def _comparison_pipeline_worker(
    dataset, user_hint: str, log_q: queue.SimpleQueue, result: dict, label: str
):
    """
    Comparison worker: does NOT touch sys.stdout so two can run truly in parallel.
    Only explicit log_q.put() calls reach the UI; pipeline print() goes to terminal.
    """
    try:
        from graph import build_graph

        log_q.put(f"[{label}] 🚀  Building pipeline…")
        app = build_graph()

        init_state = {"dataset": dataset, "user_hint": user_hint or None, "errors": []}
        log_q.put(f"[{label}] ⚙️  Starting…")

        merged:     dict = {}
        nodes_done: list = []

        for chunk in app.stream(init_state):
            node_name  = list(chunk.keys())[0]
            node_delta = chunk[node_name]
            merged.update(node_delta)
            nodes_done.append(node_name)

            log_q.put(f"[{label}] ✅  NODE COMPLETE: {node_name.upper()}")

            result["completed_nodes"] = list(nodes_done)
            result["state"]           = dict(merged)

        result["done"]  = True
        result["error"] = None
        log_q.put(f"[{label}] 🏁  All {len(nodes_done)} nodes finished.")

    except Exception as exc:
        import traceback
        err = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        result["done"]  = True
        result["error"] = err
        log_q.put(f"[{label}] ❌  PIPELINE FAILED: {err}")
    finally:
        log_q.put(_SENTINEL)


# ── Comparison rating ─────────────────────────────────────────────────────────

def _compute_comparison(ref_state: dict, syn_state: dict) -> dict:
    """Compare reference vs synthetic pipeline states and return rating info."""

    def _metrics(state):
        return (state.get("universal_metric_results", []) or []) + \
               (state.get("per_metric_results",       []) or [])

    def _avg_q(metrics):
        scores = [m.get("quality_score", 0) for m in metrics
                  if m.get("quality_score") is not None]
        return sum(scores) / len(scores) if scores else 0.0

    ref_m = _metrics(ref_state)
    syn_m = _metrics(syn_state)
    ref_avg = _avg_q(ref_m)
    syn_avg = _avg_q(syn_m)

    ratio = min(syn_avg / ref_avg, 1.0) if ref_avg > 0.001 else (0.5 if syn_avg == 0 else 1.0)

    if ratio >= 0.82:
        rating, icon, color = "ACCEPTABLE",    "🟢", "green"
    elif ratio >= 0.65:
        rating, icon, color = "ABOVE AVERAGE", "🔵", "blue"
    elif ratio >= 0.48:
        rating, icon, color = "AVERAGE",       "🟡", "orange"
    else:
        rating, icon, color = "POOR",          "🔴", "red"

    ref_by = {m.get("metric_name", ""): m for m in ref_m}
    syn_by = {m.get("metric_name", ""): m for m in syn_m}
    all_names = sorted(set(ref_by) | set(syn_by))

    per_metric = []
    for name in all_names:
        rq = (ref_by.get(name) or {}).get("quality_score")
        sq = (syn_by.get(name) or {}).get("quality_score")
        diff = (sq - rq) if (rq is not None and sq is not None) else None
        if diff is None:
            status = "—"
        elif diff >= -0.05:
            status = "✅ On par"
        elif diff >= -0.15:
            status = "🟡 Slight gap"
        elif diff >= -0.30:
            status = "🟠 Notable gap"
        else:
            status = "🔴 Large gap"
        per_metric.append({
            "Metric":       name,
            "Ref Quality":  f"{rq:.3f}" if rq is not None else "—",
            "Syn Quality":  f"{sq:.3f}" if sq is not None else "—",
            "Δ":            f"{diff:+.3f}" if diff is not None else "—",
            "Status":       status,
        })

    ref_verd = (ref_state.get("evaluator_output") or {}).get("final_verdict", "UNKNOWN")
    syn_verd = (syn_state.get("evaluator_output") or {}).get("final_verdict", "UNKNOWN")

    return {
        "ref_avg_quality":  ref_avg,
        "syn_avg_quality":  syn_avg,
        "similarity":       ratio,
        "rating":           rating,
        "icon":             icon,
        "color":            color,
        "per_metric":       per_metric,
        "ref_verdict":      ref_verd,
        "syn_verdict":      syn_verd,
        "ref_metric_count": len(ref_m),
        "syn_metric_count": len(syn_m),
    }


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_csv(f):
    df = pd.read_csv(f)
    return Dataset.from_pandas(df), df

def _load_txt(f):
    lines = [l.strip() for l in f.read().decode("utf-8", errors="ignore").splitlines() if l.strip()]
    df    = pd.DataFrame({"text": lines})
    return Dataset.from_pandas(df), df

def _load_docx(f):
    from docx import Document
    paras = [p.text.strip() for p in Document(f).paragraphs if p.text.strip()]
    df    = pd.DataFrame({"paragraph": paras})
    return Dataset.from_pandas(df), df

def _parse_hf_name(raw: str) -> str:
    """
    Accepts any of these and returns just the dataset id:
      - domenicrosati/TruthfulQA
      - load_dataset("domenicrosati/TruthfulQA")
      - from datasets import load_dataset\nds = load_dataset("domenicrosati/TruthfulQA")
      - https://huggingface.co/datasets/domenicrosati/TruthfulQA
    """
    import re
    # Extract from load_dataset("...") or load_dataset('...')
    m = re.search(r'load_dataset\s*\(\s*["\']([^"\']+)["\']', raw)
    if m:
        return m.group(1).strip()
    # Extract from HuggingFace URL
    m = re.search(r'huggingface\.co/datasets/([^\s\'"]+)', raw)
    if m:
        return m.group(1).strip().rstrip("/")
    # Fall back: take the first token that looks like a dataset id (word/word or plain word)
    m = re.search(r'([\w.\-]+/[\w.\-]+|[\w.\-]{2,})', raw.strip())
    if m:
        return m.group(1).strip()
    return raw.strip()


def _load_hf(name, split="train[:1000]"):
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    return ds, ds.to_pandas()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _arxiv_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""

def _extract_metric_value(output: str) -> str:
    for line in (output or "").splitlines():
        if line.startswith("METRIC_VALUE:"):
            return line.replace("METRIC_VALUE:", "").strip()
    return "—"

def _clean_exec_output(output: str) -> str:
    """Remove METRIC_VALUE / ANOMALOUS_ROWS lines; keep human-readable lines."""
    skip = {"METRIC_VALUE:", "ANOMALOUS_ROWS:"}
    lines = [
        l for l in (output or "").splitlines()
        if not any(l.startswith(s) for s in skip) and l.strip()
    ]
    return "\n".join(lines[:30])  # cap at 30 lines


# ── PDF generation ───────────────────────────────────────────────────────────

from pdf_report import build_pdf as _build_pdf_module

def _build_pdf(state: dict, mode: str = "full") -> bytes:
    return _build_pdf_module(state, mode=mode)



# ── Session state ─────────────────────────────────────────────────────────────

_SS: dict = {
    "dataset":     None,
    "df":          None,
    "thread":      None,
    "log_q":       None,
    "result":      None,
    "log_lines":   [],
    "done":        False,
    "error":       None,
    "final_state": None,
    "user_hint":   "",
}
for _k, _v in _SS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Comparison session state ──────────────────────────────────────────────────

_CMP_SS: dict = {
    "cmp_ref_dataset":     None,
    "cmp_ref_df":          None,
    "cmp_syn_dataset":     None,
    "cmp_syn_df":          None,
    "cmp_ref_thread":      None,
    "cmp_syn_thread":      None,
    "cmp_ref_log_q":       None,
    "cmp_syn_log_q":       None,
    "cmp_ref_result":      None,
    "cmp_syn_result":      None,
    "cmp_ref_log_lines":   [],
    "cmp_syn_log_lines":   [],
    "cmp_started":         False,   # True once Run Comparison is clicked
    "cmp_ref_done":        False,
    "cmp_syn_done":        False,
    "cmp_ref_error":       None,
    "cmp_syn_error":       None,
    "cmp_ref_final_state": None,
    "cmp_syn_final_state": None,
    "cmp_user_hint":       "",
}
for _k, _v in _CMP_SS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Safe defaults for mode-specific variables ────────────────────────────────

cmp_run_btn  = False
cmp_user_ctx = ""
run_btn      = False
user_ctx     = ""

# ── Stage labels (defined early; used in both single and comparison sections) ─

_NODE_LABELS = [
    ("extract_metadata",  "Metadata"),
    ("universal_metrics", "Universal Metrics"),
    ("thinker",           "Thinker"),
    ("researcher",        "Researcher"),
    ("evaluator",         "Evaluator"),
]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 SynAgent")

    _app_mode = st.radio(
        "Mode",
        ["Single Analysis", "Comparison"],
        key="app_mode",
        horizontal=True,
    )
    st.divider()

    # ── Single Analysis sidebar ───────────────────────────────────────────────
    if _app_mode == "Single Analysis":
        source = st.radio("Data Source", ["Upload", "HuggingFace"])

        if source == "Upload":
            uploaded = st.file_uploader("File", type=["csv", "txt", "docx"])
            if uploaded:
                ext      = Path(uploaded.name).suffix.lower()
                _loaders = {".csv": _load_csv, ".txt": _load_txt, ".docx": _load_docx}
                try:
                    if ext in _loaders:
                        ds, df = _loaders[ext](uploaded)
                        st.session_state.update({"dataset": ds, "df": df})
                        st.success(f"✓ {len(df):,} rows · {len(df.columns)} cols")
                    else:
                        st.error(f"Unsupported format: {ext}")
                except Exception as e:
                    st.error(str(e))
        else:
            hf_raw = st.text_input(
                "Dataset name or load_dataset() snippet",
                placeholder='e.g.  domenicrosati/TruthfulQA',
            )
            if hf_raw:
                hf_name = _parse_hf_name(hf_raw)
                if hf_name != hf_raw.strip():
                    st.caption(f"Parsed as: `{hf_name}`")
                try:
                    ds, df = _load_hf(hf_name)
                    st.session_state.update({"dataset": ds, "df": df})
                    st.success(f"✓ {len(df):,} rows loaded")
                except Exception as e:
                    st.error(str(e))

        user_ctx  = st.text_area("Context / Hint (optional)", height=80)

        _thread   = st.session_state.get("thread")
        _is_alive = _thread is not None and _thread.is_alive()

        run_btn = st.button(
            "▶ Run Audit",
            disabled=(st.session_state["dataset"] is None or _is_alive),
            use_container_width=True,
        )

        if _is_alive:
            _nd = (st.session_state.get("result") or {}).get("completed_nodes", [])
            st.caption(f"⏳ Running… {len(_nd)} node(s) done")

        if st.button("🔄 Reset", disabled=_is_alive, use_container_width=True):
            for k, v in _SS.items():
                st.session_state[k] = v
            st.rerun()

    # ── Comparison sidebar ────────────────────────────────────────────────────
    else:
        _cmp_started_sb = st.session_state.get("cmp_started", False)

        _cmp_ref_alive = (
            st.session_state.get("cmp_ref_thread") is not None
            and st.session_state["cmp_ref_thread"].is_alive()
        )
        _cmp_syn_alive = (
            st.session_state.get("cmp_syn_thread") is not None
            and st.session_state["cmp_syn_thread"].is_alive()
        )
        _cmp_running = _cmp_ref_alive or _cmp_syn_alive

        # File uploaders live here so dataset session state is set BEFORE
        # the button is rendered (sidebar executes top-to-bottom in one pass).
        if not _cmp_started_sb:
            st.markdown("**📂 Reference Data (Real)**")
            _sb_ref_up = st.file_uploader(
                "Upload reference dataset",
                type=["csv", "txt", "docx"],
                key="cmp_ref_upload",
            )
            if _sb_ref_up:
                _ext = Path(_sb_ref_up.name).suffix.lower()
                _ldr = {".csv": _load_csv, ".txt": _load_txt, ".docx": _load_docx}
                try:
                    if _ext in _ldr:
                        _ds, _df = _ldr[_ext](_sb_ref_up)
                        st.session_state.update({"cmp_ref_dataset": _ds, "cmp_ref_df": _df})
                        st.caption(f"✓ {len(_df):,} rows · {len(_df.columns)} cols")
                    else:
                        st.error(f"Unsupported: {_ext}")
                except Exception as _e:
                    st.error(str(_e))

            st.markdown("**🧪 Synthetic Data**")
            _sb_syn_up = st.file_uploader(
                "Upload synthetic dataset",
                type=["csv", "txt", "docx"],
                key="cmp_syn_upload",
            )
            if _sb_syn_up:
                _ext = Path(_sb_syn_up.name).suffix.lower()
                _ldr = {".csv": _load_csv, ".txt": _load_txt, ".docx": _load_docx}
                try:
                    if _ext in _ldr:
                        _ds, _df = _ldr[_ext](_sb_syn_up)
                        st.session_state.update({"cmp_syn_dataset": _ds, "cmp_syn_df": _df})
                        st.caption(f"✓ {len(_df):,} rows · {len(_df.columns)} cols")
                    else:
                        st.error(f"Unsupported: {_ext}")
                except Exception as _e:
                    st.error(str(_e))

        cmp_user_ctx = st.text_area("Context / Hint (optional)", height=80, key="cmp_hint")

        cmp_run_btn = st.button(
            "▶ Run Comparison",
            disabled=(
                st.session_state.get("cmp_ref_dataset") is None
                or st.session_state.get("cmp_syn_dataset") is None
                or _cmp_running
                or _cmp_started_sb
            ),
            use_container_width=True,
        )

        if _cmp_running:
            _ref_nd = (st.session_state.get("cmp_ref_result") or {}).get("completed_nodes", [])
            _syn_nd = (st.session_state.get("cmp_syn_result") or {}).get("completed_nodes", [])
            st.caption(f"⏳ Ref: {len(_ref_nd)} node(s) · Syn: {len(_syn_nd)} node(s)")

        if st.button("🔄 Reset Comparison", disabled=_cmp_running, use_container_width=True):
            for k, v in _CMP_SS.items():
                st.session_state[k] = [] if isinstance(v, list) else v
            st.rerun()

        pass  # cmp_run_btn / cmp_user_ctx set above; run_btn stays False


# ── Start single pipeline ─────────────────────────────────────────────────────

if run_btn:
    log_q  = queue.SimpleQueue()
    result = {"done": False, "error": None, "state": None, "completed_nodes": []}

    st.session_state.update({
        "log_lines":   [],
        "done":        False,
        "error":       None,
        "final_state": None,
        "log_q":       log_q,
        "result":      result,
        "user_hint":   user_ctx,
    })

    t = threading.Thread(
        target=_pipeline_worker,
        args=(st.session_state["dataset"], user_ctx, log_q, result),
        daemon=True,
    )
    st.session_state["thread"] = t
    t.start()
    st.rerun()


# ── Start comparison pipelines (parallel) ─────────────────────────────────────

if _app_mode == "Comparison" and cmp_run_btn:
    _ref_q   = queue.SimpleQueue()
    _syn_q   = queue.SimpleQueue()
    _ref_res = {"done": False, "error": None, "state": None, "completed_nodes": []}
    _syn_res = {"done": False, "error": None, "state": None, "completed_nodes": []}

    st.session_state.update({
        "cmp_ref_log_lines":   [],
        "cmp_syn_log_lines":   [],
        "cmp_started":         True,
        "cmp_ref_done":        False,
        "cmp_syn_done":        False,
        "cmp_ref_error":       None,
        "cmp_syn_error":       None,
        "cmp_ref_final_state": None,
        "cmp_syn_final_state": None,
        "cmp_ref_log_q":       _ref_q,
        "cmp_syn_log_q":       _syn_q,
        "cmp_ref_result":      _ref_res,
        "cmp_syn_result":      _syn_res,
        "cmp_user_hint":       cmp_user_ctx,
    })

    _t_ref = threading.Thread(
        target=_comparison_pipeline_worker,
        args=(st.session_state["cmp_ref_dataset"], cmp_user_ctx, _ref_q, _ref_res, "REF"),
        daemon=True,
    )
    _t_syn = threading.Thread(
        target=_comparison_pipeline_worker,
        args=(st.session_state["cmp_syn_dataset"], cmp_user_ctx, _syn_q, _syn_res, "SYN"),
        daemon=True,
    )
    st.session_state["cmp_ref_thread"] = _t_ref
    st.session_state["cmp_syn_thread"] = _t_syn
    _t_ref.start()
    _t_syn.start()
    st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

_app_mode = st.session_state.get("app_mode", "Single Analysis")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON MODE
# ══════════════════════════════════════════════════════════════════════════════

if _app_mode == "Comparison":
    st.title("🔀 SynAgent — Comparison Mode")
    st.caption(
        "Run the full audit pipeline on both a **reference (real)** dataset and a "
        "**synthetic** dataset simultaneously, then get a side-by-side quality report."
    )

    # ── Pre-run: dataset previews (uploaders are in the sidebar) ─────────────
    _cmp_started  = st.session_state.get("cmp_started", False)
    _cmp_ref_done = st.session_state.get("cmp_ref_done", False)
    _cmp_syn_done = st.session_state.get("cmp_syn_done", False)
    _both_done    = _cmp_ref_done and _cmp_syn_done

    if not _cmp_started:
        _ref_df_prev = st.session_state.get("cmp_ref_df")
        _syn_df_prev = st.session_state.get("cmp_syn_df")

        if _ref_df_prev is not None or _syn_df_prev is not None:
            _pc1, _pc2 = st.columns(2)
            with _pc1:
                if _ref_df_prev is not None:
                    st.markdown("**📂 Reference preview**")
                    st.dataframe(_ref_df_prev.head(10), use_container_width=True)
            with _pc2:
                if _syn_df_prev is not None:
                    st.markdown("**🧪 Synthetic preview**")
                    st.dataframe(_syn_df_prev.head(10), use_container_width=True)

        if (st.session_state.get("cmp_ref_dataset") is None
                or st.session_state.get("cmp_syn_dataset") is None):
            st.info("⬅️ Upload both datasets in the sidebar, then click **▶ Run Comparison**.")
        st.stop()

    # ── Drain both log queues every rerun ─────────────────────────────────────
    for _side, _qkey, _lkey, _rkey, _dkey, _ekey, _fskey in [
        ("REF", "cmp_ref_log_q", "cmp_ref_log_lines", "cmp_ref_result",
         "cmp_ref_done", "cmp_ref_error", "cmp_ref_final_state"),
        ("SYN", "cmp_syn_log_q", "cmp_syn_log_lines", "cmp_syn_result",
         "cmp_syn_done", "cmp_syn_error", "cmp_syn_final_state"),
    ]:
        _q   = st.session_state.get(_qkey)
        _res = st.session_state.get(_rkey)
        if _q is not None and _res is not None and not st.session_state.get(_dkey):
            _got_sent = False
            while True:
                try:
                    _ln = _q.get_nowait()
                    if _ln == _SENTINEL:
                        _got_sent = True
                        break
                    st.session_state[_lkey].append(_ln)
                except queue.Empty:
                    break
            if _got_sent:
                st.session_state[_dkey]  = True
                st.session_state[_ekey]  = _res.get("error")
                st.session_state[_fskey] = _res.get("state") or {}

    # Re-read done flags after drain
    _cmp_ref_done = st.session_state.get("cmp_ref_done", False)
    _cmp_syn_done = st.session_state.get("cmp_syn_done", False)
    _both_done    = _cmp_ref_done and _cmp_syn_done

    # ── Live progress (shown while pipelines are running) ─────────────────────
    if not _both_done:
        st.markdown("### ⚙️ Pipeline Progress")
        _prog_c1, _prog_c2 = st.columns(2)

        for _col, _label, _lkey, _rkey, _done_flag in [
            (_prog_c1, "📂 Reference", "cmp_ref_log_lines", "cmp_ref_result", _cmp_ref_done),
            (_prog_c2, "🧪 Synthetic", "cmp_syn_log_lines", "cmp_syn_result", _cmp_syn_done),
        ]:
            with _col:
                st.markdown(f"**{_label}**")
                _res   = st.session_state.get(_rkey) or {}
                _compl = list(dict.fromkeys(_res.get("completed_nodes", [])))
                _n_done  = len({n for n in _compl if n in dict(_NODE_LABELS)})
                _prog_pct = 1.0 if _done_flag else _n_done / len(_NODE_LABELS)
                _prog_txt = "✅ Done" if _done_flag else f"⏳ {_n_done}/{len(_NODE_LABELS)} nodes"
                st.progress(_prog_pct, text=_prog_txt)

                _n_cols = st.columns(len(_NODE_LABELS))
                for _i, (_key, _lbl) in enumerate(_NODE_LABELS):
                    _n_cols[_i].markdown(f"{'✅' if _key in _compl else '⬜'} {_lbl}")

                _lines = st.session_state.get(_lkey, [])
                if _lines:
                    with st.expander("Execution log ▼", expanded=False):
                        _render_detail_log(_lines, height=250)

                if not _done_flag:
                    # Show which node is actively running
                    _next_node = next(
                        (lbl for key, lbl in _NODE_LABELS if key not in _compl), None
                    )
                    if _next_node:
                        st.caption(
                            f"🔄 **{_next_node}** is running — LLM nodes can take "
                            f"5–15 min. Terminal shows live output."
                        )

        # Heartbeat so it's clear the UI is alive
        st.caption(
            f"⏱️ Polling every 2 s — last check: {time.strftime('%H:%M:%S')}"
        )

        # Poll until both finish
        time.sleep(2.0)
        st.rerun()

    # ── Results (both done) ───────────────────────────────────────────────────
    if _both_done:
        _ref_state = st.session_state.get("cmp_ref_final_state") or {}
        _syn_state = st.session_state.get("cmp_syn_final_state") or {}

        # ── Download buttons ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⬇️ Download Individual Reports")
        _dl1, _dl2 = st.columns(2)
        with _dl1:
            try:
                _ref_pdf = _build_pdf(_ref_state, mode="summary")
                st.download_button(
                    "⬇️ Reference Report (PDF)",
                    data=_ref_pdf,
                    file_name="synagent_reference_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception:
                st.download_button("⬇️ Reference Report (PDF)", data=b"",
                                   file_name="error.pdf", disabled=True,
                                   use_container_width=True)
        with _dl2:
            try:
                _syn_pdf = _build_pdf(_syn_state, mode="summary")
                st.download_button(
                    "⬇️ Synthetic Report (PDF)",
                    data=_syn_pdf,
                    file_name="synagent_synthetic_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception:
                st.download_button("⬇️ Synthetic Report (PDF)", data=b"",
                                   file_name="error.pdf", disabled=True,
                                   use_container_width=True)

        # ── Comparison report ─────────────────────────────────────────────────
        st.markdown("---")
        st.header("📊 Comparison Report")

        _cmp = _compute_comparison(_ref_state, _syn_state)

        # Rating banner
        _rating_colors = {
            "ACCEPTABLE":    ("#052e16", "#4ade80"),
            "ABOVE AVERAGE": ("#172554", "#60a5fa"),
            "AVERAGE":       ("#431407", "#fb923c"),
            "POOR":          ("#450a0a", "#f87171"),
        }
        _bg, _fg = _rating_colors.get(_cmp["rating"], ("#1e293b", "#94a3b8"))
        st.markdown(
            f'<div style="background:{_bg};border-radius:10px;padding:20px 28px;'
            f'margin-bottom:16px">'
            f'<span style="font-size:2rem;font-weight:700;color:{_fg}">'
            f'{_cmp["icon"]} {_cmp["rating"]}</span><br>'
            f'<span style="color:{_fg};opacity:0.8;font-size:0.95rem">'
            f'Synthetic data quality is <strong>{_cmp["similarity"]:.0%}</strong> '
            f'of the reference quality score — '
            f'best possible rating for synthetic data is <em>Acceptable</em>.'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        # Score overview
        _sc1, _sc2, _sc3 = st.columns(3)
        _sc1.metric("Reference Avg Quality", f"{_cmp['ref_avg_quality']:.3f}")
        _sc2.metric(
            "Synthetic Avg Quality",
            f"{_cmp['syn_avg_quality']:.3f}",
            delta=f"{_cmp['syn_avg_quality'] - _cmp['ref_avg_quality']:+.3f}",
            delta_color="normal",
        )
        _sc3.metric("Similarity Ratio", f"{_cmp['similarity']:.0%}")

        # Verdict comparison
        _vc1, _vc2 = st.columns(2)
        _v_icons = {
            "HIGH_QUALITY": "🟢", "ACCEPTABLE_QUALITY": "🟡", "POOR_QUALITY": "🔴", "UNKNOWN": "⚪"
        }
        _vc1.markdown(
            f"**Reference Verdict**  \n"
            f"{_v_icons.get(_cmp['ref_verdict'],'⚪')} "
            f"{_cmp['ref_verdict'].replace('_', ' ')}"
        )
        _vc2.markdown(
            f"**Synthetic Verdict**  \n"
            f"{_v_icons.get(_cmp['syn_verdict'],'⚪')} "
            f"{_cmp['syn_verdict'].replace('_', ' ')}"
        )

        # Per-metric comparison table
        st.markdown("#### Metric-by-Metric Comparison")
        if _cmp["per_metric"]:
            st.dataframe(
                pd.DataFrame(_cmp["per_metric"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No overlapping metrics found between the two runs.")

        # Comparison PDF download
        try:
            from pdf_report import build_comparison_pdf as _build_cmp_pdf
            _cmp_pdf_bytes = _build_cmp_pdf(_ref_state, _syn_state, _cmp)
            st.download_button(
                "⬇️ Download Comparison Report (PDF)",
                data=_cmp_pdf_bytes,
                file_name="synagent_comparison_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _pdf_err:
            st.warning(f"Could not generate comparison PDF: {_pdf_err}")

        # Detailed tabs for each run
        st.markdown("---")
        st.markdown("### 🔍 Detailed Results")
        _det_tab1, _det_tab2 = st.tabs(["📂 Reference Run", "🧪 Synthetic Run"])

        def _render_run_summary(state: dict, label: str):
            _ev  = state.get("evaluator_output", {}) or {}
            _univ = state.get("universal_metric_results", []) or []
            _pm   = state.get("per_metric_results", []) or []
            _all  = _univ + _pm

            verd = _ev.get("final_verdict", "UNKNOWN")
            _vi  = _v_icons.get(verd, "⚪")
            st.markdown(f"**Verdict:** {_vi} {verd.replace('_', ' ')}")
            if _ev.get("verdict_reasoning"):
                st.info(_ev["verdict_reasoning"])

            if _all:
                _rows = [{
                    "Metric":  m.get("metric_name", ""),
                    "Value":   _extract_metric_value(m.get("execution_output", "")),
                    "Quality": round(m.get("quality_score", 0), 3),
                    "Status":  "✅" if not m.get("error") else "⚠️",
                } for m in _all]
                st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No metrics available.")

        with _det_tab1:
            _render_run_summary(_ref_state, "Reference")
        with _det_tab2:
            _render_run_summary(_syn_state, "Synthetic")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE ANALYSIS MODE
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔍 SynAgent — Data Quality Audit")

if st.session_state["dataset"] is None:
    st.info("Upload a dataset or connect a HuggingFace dataset in the sidebar to begin.")
    st.stop()

_thread   = st.session_state.get("thread")
_log_q    = st.session_state.get("log_q")
_result   = st.session_state.get("result")
_is_alive = _thread is not None and _thread.is_alive()

with st.expander("📊 Dataset Preview", expanded=not (_is_alive or st.session_state["done"])):
    _df = st.session_state["df"]
    st.dataframe(_df.head(20), use_container_width=True)
    st.caption(f"{len(_df):,} rows × {len(_df.columns)} cols")


# ── Queue drain (every rerun) ─────────────────────────────────────────────────

if _thread is not None and _log_q is not None:
    _got_sentinel = False
    while True:
        try:
            _line = _log_q.get_nowait()
            if _line == _SENTINEL:
                _got_sentinel = True
                break
            st.session_state["log_lines"].append(_line)
        except queue.Empty:
            break

    if _got_sentinel and _result is not None:
        st.session_state["done"]        = True
        st.session_state["error"]       = _result.get("error")
        st.session_state["final_state"] = _result.get("state") or {}


_showing_live = _is_alive or (_thread is not None and not st.session_state["done"])


# ── Live view ─────────────────────────────────────────────────────────────────

if _showing_live:
    _completed = (_result.get("completed_nodes") or []) if _result else []
    _unique    = list(dict.fromkeys(_completed))
    _n_done    = len({n for n in _unique if n in dict(_NODE_LABELS)})
    _n_total   = len(_NODE_LABELS)

    _next_label = next(
        (lbl for key, lbl in _NODE_LABELS if key not in _unique), "Finishing…"
    )
    st.progress(_n_done / _n_total, text=f"⏳ Running: **{_next_label}**")

    _cols = st.columns(_n_total)
    for i, (key, lbl) in enumerate(_NODE_LABELS):
        _cols[i].markdown(f"{'✅' if key in _unique else '⬜'} {lbl}")

    st.markdown("### 📡 Live Agent Output")

    # Single human-friendly status line
    _status = _get_status_line(st.session_state["log_lines"])
    st.info(_status)

    # Full detail behind an expander — collapsed by default
    with st.expander("See detailed execution log ▼", expanded=False):
        _render_detail_log(st.session_state["log_lines"])

    # Rerun less frequently to reduce flicker (increased from 0.4s to 2s)
    if _is_alive:
        time.sleep(2.0)
        st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state["done"] and not _showing_live:

    _completed = (_result.get("completed_nodes") or []) if _result else []
    _unique    = list(dict.fromkeys(_completed))
    _cols2     = st.columns(len(_NODE_LABELS))
    for i, (key, lbl) in enumerate(_NODE_LABELS):
        _cols2[i].markdown(f"{'✅' if key in _unique else '⬜'} {lbl}")
    st.progress(1.0, text="✅ Pipeline complete")

    _err = st.session_state.get("error")
    if _err:
        st.error("❌ Pipeline error")
        with st.expander("Error details"):
            st.code(_err)

    with st.expander("📜 Full Execution Log", expanded=False):
        _render_detail_log(st.session_state["log_lines"], height=500)

    _state = st.session_state.get("final_state") or {}
    if not _state:
        st.warning("Pipeline returned no state — see log for details.")
        st.stop()

    _thinker    = _state.get("thinker_output", {})    or {}
    _researcher = _state.get("researcher_output", {}) or {}
    _evaluator  = _state.get("evaluator_output", {})  or {}
    _per_metric = _state.get("per_metric_results", []) or []
    _universal  = _state.get("universal_metric_results", []) or []

    st.markdown("---")
    st.header("📋 Audit Results")

    # ── Download buttons ──────────────────────────────────────────────────────
    _dl_col1, _dl_col2, _dl_col3 = st.columns(3)

    with _dl_col1:
        try:
            _pdf_summary = _build_pdf(_state, mode="summary")
            st.download_button(
                label="⬇️ Summary (PDF)",
                data=_pdf_summary,
                file_name="synagent_summary_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _e:
            st.download_button("⬇️ Summary (PDF)", data=b"", file_name="error.pdf",
                               disabled=True, use_container_width=True)

    with _dl_col2:
        _universal_json = json.dumps(_universal, indent=2, default=str)
        st.download_button(
            label="⬇️ Universal Metrics",
            data=_universal_json,
            file_name="universal_metrics.json",
            mime="application/json",
            use_container_width=True,
        )

    with _dl_col3:
        _research_metrics = [r for r in _per_metric if r.get("metric_source") != "universal"]
        _research_json = json.dumps(_research_metrics, indent=2, default=str)
        st.download_button(
            label="⬇️ Research Metrics",
            data=_research_json,
            file_name="research_metrics.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Thinker", "🔬 Researcher", "📊 Metrics", "🏆 Verdict"])

    # ── Thinker ───────────────────────────────────────────────────────────────

    with tab1:
        if not _thinker:
            st.info("No thinker output.")
        else:
            # User intent — show first so it's clear this shaped everything below
            _hint = st.session_state.get("user_hint", "").strip()
            _influence = _thinker.get("user_hint_influence", "").strip()
            if _hint:
                st.markdown("#### 🎯 Your Intent")
                st.markdown(
                    f'<div style="border-left:4px solid #6366f1;padding:10px 16px;'
                    f'background:#1e1b4b;border-radius:4px;color:#c7d2fe;font-style:italic">'
                    f'"{_hint}"</div>',
                    unsafe_allow_html=True,
                )
                if _influence:
                    st.markdown(
                        f'<div style="margin-top:8px;padding:8px 14px;'
                        f'background:#0f172a;border-radius:4px;color:#94a3b8;font-size:13px">'
                        f'💡 <strong>How it shaped the analysis:</strong> {_influence}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("")
            elif _influence:
                st.caption(f"💡 Hint influence: {_influence}")

            _dom = _thinker.get("domain", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Domain",       _dom.get("name", "—"))
            c2.metric("Dataset Type", _thinker.get("dataset_type", "—"))
            c3.metric("Confidence",   f"{_dom.get('confidence', 0):.0%}")

            if _dom.get("reasoning"):
                st.caption(_dom["reasoning"])

            # Reasoning trace
            _trace = _thinker.get("reasoning_trace", "")
            if _trace:
                st.markdown("#### 💭 Reasoning Trace")
                st.info(_trace)

            # Surface quality hints
            _hints = _thinker.get("surface_quality_hints", [])
            if _hints:
                st.markdown("#### 🔎 Surface Quality Hints")
                for h in _hints:
                    st.markdown(f"- {h}")

            # Security recommendations (previously governance_warnings)
            _sec = _thinker.get("governance_warnings", [])
            if _sec:
                st.markdown("#### 🔒 Security Recommendations")
                for w in _sec:
                    st.warning(w)

            # Recommended analysis types
            _agents = _thinker.get("recommended_agents", {})
            if _agents:
                _analysis_labels = {
                    "governance": "Data Governance & Privacy Review",
                    "math":       "Mathematical & Statistical Analysis",
                    "semantic":   "Semantic & Text Quality Analysis",
                    "utility":    "Utility & Downstream Task Evaluation",
                }
                st.markdown("#### 🔬 Recommended Analysis Types")
                for key, info in _agents.items():
                    label = _analysis_labels.get(key, key.replace("_", " ").title())
                    if info.get("run"):
                        st.success(f"**{label}** — {info.get('reason', '')}")
                    else:
                        st.markdown(f"⏭️ **{label}** — not recommended. {info.get('reason', '')}")

            # Column profiles
            _col_profiles = _thinker.get("column_profiles", [])
            if _col_profiles:
                st.markdown("#### 📋 Column Profiles")
                _rows = [{
                    "Column": c.get("column_name", ""),
                    "Type":   c.get("inferred_dtype", ""),
                    "Role":   c.get("semantic_role", ""),
                    "Notes":  c.get("notes", "")[:100],
                } for c in _col_profiles]
                st.dataframe(pd.DataFrame(_rows), use_container_width=True)

            # Execution notes
            _exec_notes = _thinker.get("execution_notes", [])
            if _exec_notes:
                st.markdown("#### 📝 Execution Notes")
                for n in _exec_notes:
                    st.markdown(f"- {n}")

    # ── Researcher ────────────────────────────────────────────────────────────

    with tab2:
        if not _researcher:
            st.info("No researcher output.")
        else:
            _metrics_list = _researcher.get("final_metrics") or _researcher.get("proposed_metrics", [])
            _summary      = _researcher.get("research_summary", "")
            _confidence   = _researcher.get("confidence_level", "")

            if _confidence:
                _conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(_confidence, "gray")
                st.markdown(f"**Research Confidence:** :{_conf_color}[{_confidence}]")

            if _summary:
                st.info(_summary)

            st.subheader(f"📌 {len(_metrics_list)} Proposed Metrics")

            if _metrics_list:
                # Summary table
                _tbl = pd.DataFrame([{
                    "Name":      m.get("metric_name", ""),
                    "Type":      m.get("metric_type", ""),
                    "Relevance": round(m.get("relevance_score", 0), 2),
                    "Papers":    len(m.get("paper_citations", [])),
                } for m in _metrics_list])
                st.dataframe(_tbl, use_container_width=True)

                st.markdown("---")

                # Per-metric detail
                for m in _metrics_list:
                    with st.expander(f"📌 {m.get('metric_name', '?')}  ·  {m.get('metric_type','')}"):
                        st.markdown(f"**{m.get('description', '')}**")
                        st.markdown(m.get("reasoning", ""))

                        _hint = m.get("execution_hint", "")
                        if _hint:
                            st.markdown("**How to compute:**")
                            st.code(_hint, language="python")

                        _cites = m.get("paper_citations", [])
                        if _cites:
                            st.markdown("**Supporting Research:**")
                            for c in _cites:
                                _aid   = c.get("arxiv_id", "")
                                _title = c.get("title", "Untitled")
                                _year  = c.get("year", "")
                                _auth  = c.get("authors", "")
                                _sup   = c.get("supporting_text", "")[:160]

                                if _aid:
                                    st.markdown(
                                        f"📄 [{_title}]({_arxiv_url(_aid)}) — {_auth} ({_year})"
                                    )
                                else:
                                    st.markdown(f"📄 *{_title}* — {_auth} ({_year})")

                                if _sup:
                                    st.caption(f'"{_sup}"')

    # ── Metrics ───────────────────────────────────────────────────────────────

    with tab3:
        _all_metrics = (_universal or []) + (_per_metric or [])
        if not _all_metrics:
            st.info("No metrics computed.")
        else:
            st.subheader(f"📊 {len(_all_metrics)} Metrics Computed")

            # Total rows for percentage calculation
            _total_rows = max(_state.get("raw_metadata", {}).get("row_count", 1) or 1, 1)

            def _anom_pct(count):
                pct = count / _total_rows * 100
                return f"{pct:.1f}% ({count} rows)"

            # Summary table
            _sum_rows = []
            for r in _all_metrics:
                _sum_rows.append({
                    "Metric":    r.get("metric_name", ""),
                    "Type":      r.get("metric_type", ""),
                    "Value":     _extract_metric_value(r.get("execution_output", "")),
                    "Anomalies": _anom_pct(r.get("anomalous_row_count", 0)),
                    "Quality":   round(r.get("quality_score", 0), 2),
                    "Status":    "✅" if not r.get("error") else "⚠️",
                })
            st.dataframe(pd.DataFrame(_sum_rows), use_container_width=True)

            st.markdown("---")

            for r in _all_metrics:
                _mval  = _extract_metric_value(r.get("execution_output", ""))
                _icon  = "✅" if not r.get("error") else "⚠️"
                _label = f"{_icon} {r.get('metric_name','')}  ·  value: **{_mval}**"

                with st.expander(_label):
                    # Header row
                    _mc1, _mc2, _mc3 = st.columns(3)
                    _mc1.markdown(f"**Type**  \n{r.get('metric_type','—')}")
                    _mc2.markdown(f"**Strategy**  \n{r.get('computation_strategy', r.get('metric_source','—'))}")
                    _mc3.markdown(f"**Anomalous rows**  \n{_anom_pct(r.get('anomalous_row_count', 0))}")

                    # Interpretation — primary output
                    _interp = r.get("metric_interpretation", "")
                    if _interp:
                        st.markdown("**Interpretation**")
                        st.info(_interp)

                    # Cleaned execution output (no raw METRIC_VALUE lines)
                    _clean = _clean_exec_output(r.get("execution_output", ""))
                    if _clean:
                        with st.expander("Computation details"):
                            st.code(_clean, language="text")

                    # Anomalous row samples
                    _anoms = r.get("anomalous_row_narratives", [])
                    if _anoms:
                        _anom_count = r.get("anomalous_row_count", 0)
                        with st.expander(f"Flagged rows — {_anom_pct(_anom_count)}"):
                            for n in _anoms[:10]:
                                st.caption(n)

                    # Error
                    if r.get("error"):
                        st.error(f"Error: {r['error']}")

                    # Generated code (collapsed)
                    _code = r.get("generated_code", "")
                    _skip = {"# LLM-based semantic scoring", "# Tool registry (pre-built or cached)", ""}
                    if _code and _code not in _skip:
                        with st.expander("Generated code"):
                            st.code(_code, language="python")

    # ── Verdict ───────────────────────────────────────────────────────────────

    with tab4:
        if not _evaluator:
            st.info("No evaluator output yet.")
        else:
            _verdict  = _evaluator.get("final_verdict", "UNKNOWN")
            _v_icons  = {"HIGH_QUALITY": "🟢", "ACCEPTABLE_QUALITY": "🟡", "POOR_QUALITY": "🔴"}
            _v_colors = {"HIGH_QUALITY": "green", "ACCEPTABLE_QUALITY": "orange", "POOR_QUALITY": "red"}

            st.markdown(
                f"## {_v_icons.get(_verdict,'⚪')} "
                f":{_v_colors.get(_verdict,'gray')}[{_verdict.replace('_',' ')}]"
            )

            _vreason = _evaluator.get("verdict_reasoning", "")
            if _vreason:
                st.info(_vreason)

            _evidence = _evaluator.get("dataset_level_evidence", "")
            if _evidence:
                st.markdown("#### Dataset-Level Evidence")
                st.markdown(_evidence)

            _obs = _evaluator.get("quality_observations", "")
            if _obs:
                st.markdown("#### Quality Observations")
                st.markdown(_obs)

            _stat = _evaluator.get("statistical_justification", "")
            if _stat:
                st.markdown("#### Statistical Justification")
                st.markdown(_stat)

            _by_metric = _evaluator.get("quality_by_metric", [])
            if _by_metric:
                st.markdown("#### Per-Metric Quality")
                _lv = {"GOOD": "🟢", "ACCEPTABLE": "🟡", "POOR": "🔴"}
                _vrows = [{
                    "Metric":  m.get("metric", ""),
                    "Level":   f"{_lv.get(m.get('quality_level',''),'⚪')} {m.get('quality_level','')}",
                    "Finding": m.get("finding", "")[:150],
                    "Note":    m.get("note", ""),
                } for m in _by_metric]
                st.dataframe(pd.DataFrame(_vrows), use_container_width=True)
