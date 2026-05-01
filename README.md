# SynAgent — Data Quality Audit

An agentic pipeline that audits datasets for quality using a LangGraph multi-agent system (Thinker → Researcher → Evaluator), with a Streamlit UI for live streaming results.

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/kashcentific/SynAgent-Final.git
cd SynAgent-Final
pip install -r requirements.txt
```

### API Key

This project uses [OpenRouter](https://openrouter.ai) to access LLMs (default: `gpt-4o-mini`).

1. Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Create a `.env` file in the project root:

```bash
cp .env.example .env
```

3. Edit `.env` and paste your key:

```
OPENAI_API_KEY=sk-or-...
```

## Running

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

## Usage

**Single Analysis** — Upload a CSV, TXT, or DOCX file (or enter a HuggingFace dataset name) and click **Run Audit**. The pipeline runs 5 agents and produces a quality report with PDF export.

**Comparison Mode** — Upload a reference (real) dataset and a synthetic dataset side-by-side to get a similarity rating and per-metric comparison.

## Pipeline

```
extract_metadata → universal_metrics → thinker → researcher → evaluator
```

| Agent | What it does |
|---|---|
| Metadata | Column stats, row counts, type inference |
| Universal Metrics | Baseline text quality scores (NLTK, sklearn) |
| Thinker | Identifies domain, dataset type, quality profile via LLM |
| Researcher | Searches ArXiv + web for relevant quality metrics |
| Evaluator | Runs metrics, scores quality, produces final verdict |
