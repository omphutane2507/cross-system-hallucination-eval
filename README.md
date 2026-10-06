# Hallucination Detection on Resource-Constrained Hardware — Starter Project

This is the starter codebase for your paper:
"Accuracy-Efficiency Trade-offs in Hallucination Detection Methods for
LLM Summaries on Resource-Constrained Hardware"

## What this does
1. Downloads FaithBench (real hallucination annotations on summaries from
   10 modern LLMs including LLaMA-3, Mistral, GPT-4o, Claude, etc.)
2. Cleans it into a simple CSV with binary hallucination labels
3. Runs two lightweight detectors (NLI-based, retrieval-based) and
   measures BOTH accuracy (Precision/Recall/F1) AND resource cost
   (GPU memory, latency) -- this combination is your paper's novelty.

## How to run this on YOUR machine (RTX 3050 4GB)

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline, in order
python scripts/01_download_data.py
python scripts/02_prepare_dataset.py
python scripts/03_run_detectors.py
```

Expected runtime on a 4GB GPU with MAX_SAMPLES=300: roughly 15-30 minutes
total, depending on your CPU/disk speed. If you hit an out-of-memory error,
lower MAX_SAMPLES in scripts/03_run_detectors.py, or truncate source/summary
text further (already truncated to 1024/512 tokens as a safeguard).

## What you get out

`results/detector_comparison.csv` -- your core Results table, with columns:
- precision, recall, f1 (accuracy)
- avg_latency_ms (speed)
- peak_memory_mb (resource cost)

This single table is the evidence for your paper's central claim: which
detector gives the best accuracy-per-resource trade-off.

## Next steps to extend this into a full paper's worth of results
1. Run with different threshold values for each detector (e.g., 0.3, 0.4,
   0.5, 0.6 for NLI entailment cutoff) -- gives you an ablation table.
2. Break results down BY model_name (df.groupby("model_name")) to see if
   detectors perform differently on LLaMA-3 vs GPT-4o vs Mistral outputs
   -- this is a nice additional figure.
3. Manually inspect 15-20 false positives/negatives and categorize the
   error types (numeric, entity, event) for your Discussion section.
4. Increase MAX_SAMPLES once you've confirmed your GPU handles it, for a
   more statistically solid result.

## Citing your data source
Cite the FaithBench paper in your Related Work / Methodology section:
Bao et al., "FaithBench: A Diverse Hallucination Benchmark for
Summarization by Modern LLMs," NAACL 2025.
Repo: https://github.com/vectara/FaithBench
