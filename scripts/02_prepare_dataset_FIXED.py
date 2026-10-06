"""
Step 2 (FIXED): Parse raw FaithBench batch JSONs into a single clean CSV.

BUG FIX (important): FaithBench's real annotation labels look like:
    ["Unwanted", "Unwanted.Instrinsic"]   <-- note the typo "Instrinsic"
    ["Unwanted", "Unwanted.Extrinsic"]
    ["Questionable", "Questionable.Instrinsic"], etc.
The original version of this script matched on the correctly-spelled
"intrinsic", which never matched FaithBench's real (misspelled) labels.
That silently labeled EVERY sample as non-hallucinated (is_hallucinated=0),
which is why precision/recall/F1 all came out as exactly 0.0 -- there were
no positive examples in the ground truth at all.

Fix: match case-insensitively on the substring "unwanted" or "questionable"
instead of an exact string set, so typos/casing variants don't break it.
"Benign" labels are excluded on purpose: a benign hallucination doesn't
change factual meaning, so it shouldn't count as a true hallucination here.

Output columns:
    sample_id, source, summary, model_name, is_hallucinated (0/1), num_annotations
"""

import os
import json
import glob
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faithbench_clean.csv")

POSITIVE_SUBSTRINGS = ("unwanted", "questionable")
EXCLUDE_SUBSTRING = "benign"


def sample_is_hallucinated(sample: dict) -> int:
    annotations = sample.get("annotations", []) or []
    for ann in annotations:
        labels = ann.get("label", [])
        if isinstance(labels, str):
            labels = [labels]
        for lbl in labels:
            lbl_lower = str(lbl).lower()
            if EXCLUDE_SUBSTRING in lbl_lower:
                continue
            if any(sub in lbl_lower for sub in POSITIVE_SUBSTRINGS):
                return 1
    return 0


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(RAW_DIR, "batch_*.json")))
    if not files:
        print(f"No batch files found in {RAW_DIR}. Run 01_download_data.py first.")
        return

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            batch = json.load(f)
        for sample in batch.get("samples", []):
            rows.append({
                "sample_id": sample.get("sample_id"),
                "source": sample.get("source", ""),
                "summary": sample.get("summary", ""),
                "model_name": sample.get("summarizer", sample.get("model", "unknown")),
                "is_hallucinated": sample_is_hallucinated(sample),
                "num_annotations": len(sample.get("annotations", []) or []),
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["sample_id", "summary"])
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df)} samples to {OUT_PATH}")
    print("\nLabel balance (should now show BOTH 0 and 1, not just 0):")
    print(df["is_hallucinated"].value_counts())
    print("\nSamples per model:")
    print(df["model_name"].value_counts())


if __name__ == "__main__":
    main()