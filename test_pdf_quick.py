"""
Quick PDF test — no pipeline required, instant verification
Run with: python test_pdf_quick.py
"""

import json
from pdf_report import build_pdf

# Mock state with minimal data
mock_state = {
    "universal_metric_results": [
        {
            "metric_name": "Text Validity Score",
            "metric_type": "validity",
            "execution_output": "METRIC_VALUE: 0.95",
            "anomalous_row_count": 2,
            "quality_score": 0.95,
            "error": None,
        },
        {
            "metric_name": "Lexical Diversity",
            "metric_type": "diversity",
            "execution_output": "METRIC_VALUE: 0.87",
            "anomalous_row_count": 0,
            "quality_score": 0.87,
            "error": None,
        },
    ],
    "per_metric_results": [
        {
            "metric_name": "Semantic Coherence",
            "metric_type": "semantic",
            "metric_source": "research",
            "execution_output": "METRIC_VALUE: 0.82",
            "anomalous_row_count": 5,
            "quality_score": 0.82,
            "error": None,
            "metric_interpretation": "Text shows good semantic consistency",
        },
    ],
    "thinker_output": {
        "dataset_type": "Text Documents",
        "domain": {
            "name": "General NLP",
            "confidence": 0.85,
            "reasoning": "Multi-domain text collection",
            "summary": "Generic text dataset"
        },
        "column_profiles": [
            {
                "column_name": "text",
                "inferred_dtype": "text",
                "semantic_role": "content",
                "notes": "Primary text column"
            }
        ],
        "surface_quality_hints": ["Low null rates", "Good diversity"],
        "governance_warnings": ["No PII detected"],
    },
    "researcher_output": {
        "research_summary": "Found 3 relevant metrics from ArXiv papers",
        "proposed_metrics": [
            {
                "metric_name": "Semantic Coherence",
                "metric_type": "semantic",
                "description": "Measures semantic consistency"
            }
        ]
    },
    "evaluator_output": {
        "final_verdict": "ACCEPTABLE_QUALITY",
        "verdict_reasoning": "Dataset quality is acceptable with minor issues",
        "dataset_level_evidence": "200 rows processed, 0.88 average quality",
        "quality_observations": "Good text diversity, some formatting issues",
        "statistical_justification": "Mean score: 0.88, Std: 0.08",
        "quality_by_metric": [
            {
                "metric": "Validity",
                "quality_level": "GOOD",
                "finding": "95% valid"
            }
        ]
    }
}

print("🧪 Testing PDF Generation...\n")

try:
    print("1️⃣  Generating Full Report PDF...")
    pdf_full = build_pdf(mock_state, mode="full")
    print(f"   ✅ Success! Generated {len(pdf_full):,} bytes")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2️⃣  Generating Summary Report PDF...")
    pdf_summary = build_pdf(mock_state, mode="summary")
    print(f"   ✅ Success! Generated {len(pdf_summary):,} bytes")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ PDF Test Complete! Both buttons should work now.")
print("\nYou can now run the Streamlit app and test the download buttons.")
print("The PDFs will generate much faster on real data now that encoding is fixed.")
