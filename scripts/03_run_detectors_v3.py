"""
Step 3 (v3): Full experiment runner.

What's new vs v2:
  1. Explicit CUDA diagnostics printed at the top (so you immediately see
     if your GPU wasn't detected, and why).
  2. SYSTEM_LABEL: tag every result row with which machine produced it
     (e.g. "RTX3050", "Colab-T4", "CPU-only"). Set this before each run.
  3. MAX_SAMPLES = None runs on the FULL cleaned dataset. Set to an int
     (e.g. 300) to subsample for a quick test run first.
  4. Third detector added: a lightweight SelfCheckGPT-style self-consistency
     check. It generates K stochastic summaries of the source with a small
     local model, then checks whether each sentence of the summary being
     evaluated is supported (via NLI entailment) by at least one of those
     samples. Low average support -> flagged as hallucinated.
  5. Retrieval threshold sweep kept from v2.

Recommended usage:
  - First run with MAX_SAMPLES = 50 to sanity check everything works and
    time one full pass, THEN switch to None (full dataset) or a larger
    number once you know how long it takes on your hardware.
"""

import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================
# CONFIG -- change these per run
# ============================================================
SYSTEM_LABEL = "RTX3050"          # change to "Colab-T4" / "CPU-only" etc. per machine
MAX_SAMPLES = None                  # start small; set to None for full dataset
RETRIEVAL_THRESHOLDS = [0.55, 0.65, 0.75, 0.85, 0.90]
SELFCHECK_NUM_SAMPLES = 3          # how many stochastic summaries to generate per source
SELFCHECK_THRESHOLD = 0.5          # avg entailment below this -> flagged hallucinated
GENERATOR_MODEL = "sshleifer/distilbart-cnn-12-6"  # small, ~300MB summarization model
# ============================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faithbench_clean.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def print_cuda_diagnostics():
    print("=" * 60)
    print("CUDA DIAGNOSTICS")
    print("=" * 60)
    print(f"System label for this run : {SYSTEM_LABEL}")
    print(f"PyTorch version           : {torch.__version__}")
    print(f"CUDA available            : {torch.cuda.is_available()}")
    print(f"CUDA version (torch built): {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU device name           : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU total memory          : {total_mem:.2f} GB")
    else:
        print("No GPU detected. If you expect one, your torch install is")
        print("likely CPU-only. Fix with:")
        print("  pip uninstall torch -y")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 60 + "\n")


AVAILABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def reset_memory(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device):
    if device == "cuda":
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    return "n/a (CPU)"


def evaluate(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------
# Detector A: NLI-based
# ---------------------------------------------------------------------
def run_nli_detector(df, device):
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    reset_memory(device)
    preds, latencies = [], []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"NLI [{device}]"):
            t0 = time.perf_counter()
            inputs = tokenizer(
                row["source"][:1024], row["summary"][:512],
                truncation=True, max_length=1024, return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            entailment_score = probs[2].item()
            preds.append(1 if entailment_score < 0.5 else 0)
            latencies.append((time.perf_counter() - t0) * 1000)

    mem = get_peak_memory_mb(device)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return preds, sum(latencies) / len(latencies), mem


# ---------------------------------------------------------------------
# Detector B: Retrieval / embedding similarity-based
# ---------------------------------------------------------------------
def run_retrieval_detector(df, device, threshold):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    reset_memory(device)
    preds, latencies = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Retrieval [{device}, thr={threshold}]"):
        t0 = time.perf_counter()
        emb_source = model.encode(row["source"][:2000], convert_to_tensor=True)
        emb_summary = model.encode(row["summary"][:1000], convert_to_tensor=True)
        sim = util.cos_sim(emb_source, emb_summary).item()
        preds.append(1 if sim < threshold else 0)
        latencies.append((time.perf_counter() - t0) * 1000)

    mem = get_peak_memory_mb(device)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return preds, sum(latencies) / len(latencies), mem


# ---------------------------------------------------------------------
# Detector C: SelfCheckGPT-style self-consistency
# ---------------------------------------------------------------------
def split_sentences(text):
    # Simple sentence splitter -- good enough for this purpose, avoids
    # pulling in a heavier NLP library just for sentence boundaries.
    parts = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")]
    return [p for p in parts if len(p) > 0]


def run_selfcheck_detector(df, device):
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL).to(device)
    gen_model.eval()

    nli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to(device)
    nli_model.eval()

    reset_memory(device)
    preds, latencies = [], []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"SelfCheck [{device}]"):
            t0 = time.perf_counter()

            src_inputs = gen_tokenizer(
                row["source"][:1024], truncation=True, max_length=1024,
                return_tensors="pt"
            ).to(device)
            sampled_summaries = []
            for _ in range(SELFCHECK_NUM_SAMPLES):
                out_ids = gen_model.generate(
                    **src_inputs, do_sample=True, top_p=0.9, temperature=1.0,
                    max_new_tokens=80,
                )
                sampled_summaries.append(
                    gen_tokenizer.decode(out_ids[0], skip_special_tokens=True)
                )

            target_sentences = split_sentences(row["summary"]) or [row["summary"]]
            sentence_scores = []
            for sent in target_sentences:
                best_entailment = 0.0
                for sample in sampled_summaries:
                    nli_inputs = nli_tokenizer(
                        sample[:512], sent[:256], truncation=True,
                        max_length=512, return_tensors="pt"
                    ).to(device)
                    logits = nli_model(**nli_inputs).logits
                    probs = torch.softmax(logits, dim=1)[0]
                    best_entailment = max(best_entailment, probs[2].item())
                sentence_scores.append(best_entailment)

            avg_support = sum(sentence_scores) / len(sentence_scores)
            preds.append(1 if avg_support < SELFCHECK_THRESHOLD else 0)
            latencies.append((time.perf_counter() - t0) * 1000)

    mem = get_peak_memory_mb(device)
    del gen_model, nli_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return preds, sum(latencies) / len(latencies), mem


def main():
    print_cuda_diagnostics()

    if not os.path.exists(DATA_PATH):
        print(f"Missing {DATA_PATH}. Run 02_prepare_dataset.py first.")
        return

    df = pd.read_csv(DATA_PATH).dropna(subset=["source", "summary"])
    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)
    y_true = df["is_hallucinated"].tolist()
    print(f"Running on {len(df)} samples. Devices this run: {AVAILABLE_DEVICES}\n")
    print(f"Ground-truth label balance:\n{df['is_hallucinated'].value_counts()}\n")

    rows = []

    for device in AVAILABLE_DEVICES:
        preds, latency, mem = run_nli_detector(df, device)
        rows.append({
            "system": SYSTEM_LABEL, "detector": "NLI-based", "device": device,
            "threshold": 0.5, **evaluate(y_true, preds),
            "avg_latency_ms": round(latency, 2), "peak_memory_mb": mem,
        })

    for device in AVAILABLE_DEVICES:
        for thr in RETRIEVAL_THRESHOLDS:
            preds, latency, mem = run_retrieval_detector(df, device, thr)
            rows.append({
                "system": SYSTEM_LABEL, "detector": "Retrieval-based", "device": device,
                "threshold": thr, **evaluate(y_true, preds),
                "avg_latency_ms": round(latency, 2), "peak_memory_mb": mem,
            })

    for device in AVAILABLE_DEVICES:
        preds, latency, mem = run_selfcheck_detector(df, device)
        rows.append({
            "system": SYSTEM_LABEL, "detector": "SelfCheck-style", "device": device,
            "threshold": SELFCHECK_THRESHOLD, **evaluate(y_true, preds),
            "avg_latency_ms": round(latency, 2), "peak_memory_mb": mem,
        })

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, f"detector_comparison_{SYSTEM_LABEL}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(result_df.to_string(index=False))
    print("\nRun this same script on each machine/system with a different")
    print("SYSTEM_LABEL set at the top, then combine all the output CSVs")
    print("(they share the same columns) into one master table for the paper.")


if __name__ == "__main__":
    main()
