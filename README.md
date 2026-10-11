# Cross-System Hallucination Detection Benchmark

Accuracy-efficiency trade-offs of three hallucination detection methods for
LLM-generated summaries, measured on the **FaithBench** benchmark across a
consumer GPU, a free cloud GPU (Colab), and a different-generation cloud GPU
(Kaggle).

Most hallucination-detection work reports only accuracy. This project reports
accuracy **and** latency / GPU memory, on several hardware setups, to answer a
practical question: *which detector can you actually run on the hardware you
have?*

> Working title: *Accuracy-Efficiency Trade-offs in Hallucination Detection for
> LLM-Generated Summaries Across Heterogeneous GPU Hardware*

---

## 1. What this repo contains

```
cross-system-hallucination-eval/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 01_download_data.py            # downloads FaithBench batch JSON files
│   ├── 02_prepare_dataset_FIXED.py    # cleans them into data/faithbench_clean.csv
│   └── 03_run_detectors_v3.py         # runs the 3 detectors, writes results CSV
├── data/
│   ├── raw/                           # FaithBench batch_*.json
│   └── faithbench_clean.csv           # source, summary, is_hallucinated (0/1)
└── results/
    ├── detector_comparison_RTX3050.csv
    ├── detector_comparison_Colab-T4.csv
    └── detector_comparison_Kaggle-P100.csv
```

The scripts must be run in numeric order. Script names carry the suffixes
`_FIXED` and `_v3` because earlier versions had bugs or fewer features (see
section 8).

---

## 2. Dataset

**FaithBench** (Bao et al., NAACL 2025) provides summaries written by modern
LLMs together with human annotations of hallucinated spans. Source:
<https://github.com/vectara/FaithBench>.

| Property | Value |
|---|---|
| Samples after cleaning | 750 |
| Labelled hallucinated (1) | 511 (68.1%) |
| Labelled faithful (0) | 239 (31.9%) |

The counts above are derived from the results files (every recall value is a
whole multiple of 1/511 and every precision value implies whole-number
prediction counts out of 750), and the Colab progress bar showed 750 samples.
Confirm them by running the label-balance printout in `02_prepare_dataset_FIXED.py`.

**Labelling rule** (`02_prepare_dataset_FIXED.py`): a summary is labelled
hallucinated (1) if any annotation label contains `unwanted` or `questionable`
(case-insensitive), and labels containing `benign` are ignored. Matching is by
substring because FaithBench's own labels contain a typo (`Instrinsic`), which
silently broke the original exact-match version (section 8).

---

## 3. Detectors under test

All three are binary classifiers: output 1 = "summary is hallucinated".

| Detector | Models | Decision rule |
|---|---|---|
| **NLI-based** | `facebook/bart-large-mnli` | Source is the premise, summary the hypothesis. Flag if entailment probability < 0.5. Source truncated to 1024 characters, summary to 512. |
| **Retrieval-based** | `sentence-transformers/all-MiniLM-L6-v2` | Cosine similarity between source and summary embeddings. Flag if similarity < threshold. Thresholds swept: 0.55, 0.65, 0.75, 0.85, 0.90. |
| **SelfCheck-style** | `sshleifer/distilbart-cnn-12-6` + `facebook/bart-large-mnli` | Sample K=3 alternative summaries of the source (`do_sample=True`, `top_p=0.9`, `temperature=1.0`, `max_new_tokens=80`, `num_beams=1`). For each sentence of the evaluated summary, take its best entailment score against the K samples, then average over sentences. Flag if average < 0.5. |

This is a simplified, SelfCheckGPT-inspired method, not the original
implementation.

For every run the script records Precision, Recall, F1, **average per-sample
latency (ms)** and **peak GPU memory (MB)** (`torch.cuda.max_memory_allocated`).
Samples are processed one at a time (batch size 1).

---

## 4. Systems tested and their labels

Every result row carries a `system` label (set via `SYSTEM_LABEL` at the top of
`03_run_detectors_v3.py`) and a `device` column. Result files are named
`detector_comparison_<SYSTEM_LABEL>.csv`.

| `system` label | Platform | GPU | Devices run | Notes |
|---|---|---|---|---|
| `RTX3050` | Local Windows machine, Python venv | NVIDIA GeForce RTX 3050 (`nvidia-smi` reported 6144 MiB VRAM) | `cpu` **and** `cuda` | Driver 591.66 (reports CUDA 13.1), PyTorch 2.14.0+cu130. This is the only system with a CPU baseline. |
| `Colab-T4` | Google Colab, free tier | NVIDIA T4 (16 GB) | `cuda` only | CPU pass skipped. `num_beams=1` patch applied. |
| `Kaggle-P100` | Kaggle Notebook, free tier | NVIDIA P100 (16 GB) | `cuda` only | CPU pass skipped. `num_beams=1` patch applied. |

Not recorded (add before publishing): the CPU model of the local machine, and the
Python / PyTorch / Transformers versions on Colab and Kaggle. Run `pip freeze`
in each environment and save the output next to the results.

AWS EC2 runs (T4, A10G, V100) were planned and scripted but **never executed**
(no credits), so no AWS results are reported.

---

## 5. Setup and step-by-step usage

### 5.1 Local machine (Windows, NVIDIA GPU)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA build of PyTorch **before** running anything, otherwise `pip`
may give you the CPU-only wheel and `torch.cuda.is_available()` will be `False`.
Check the CUDA version in the header of `nvidia-smi`, then use the matching
selector on <https://pytorch.org/get-started/locally/>. For driver 591.66
(CUDA 13.1) this was the `cu130` build:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then run the pipeline:

```powershell
python scripts/01_download_data.py
python scripts/02_prepare_dataset_FIXED.py
python scripts/03_run_detectors_v3.py
```

Before the last step, edit the config block at the top of
`03_run_detectors_v3.py`:

```python
SYSTEM_LABEL = "RTX3050"   # label for this machine
MAX_SAMPLES = None         # None = all 750 samples; use 20 for a quick test
```

### 5.2 Google Colab

1. Runtime -> Change runtime type -> **T4 GPU** (before running anything).
2. In a cell:

```python
!git clone https://github.com/<your-username>/cross-system-hallucination-eval.git
%cd cross-system-hallucination-eval
!pip install -q -r requirements.txt

# label this run, GPU-only, and fix the SelfCheck slowdown
!sed -i 's/^SYSTEM_LABEL = .*/SYSTEM_LABEL = "Colab-T4"/' scripts/03_run_detectors_v3.py
!sed -i 's/^AVAILABLE_DEVICES = .*/AVAILABLE_DEVICES = ["cuda"] if torch.cuda.is_available() else ["cpu"]/' scripts/03_run_detectors_v3.py
!sed -i 's/max_new_tokens=80,$/max_new_tokens=80, num_beams=1,/' scripts/03_run_detectors_v3.py
!python scripts/03_run_detectors_v3.py
```

3. Download `results/detector_comparison_Colab-T4.csv` before the session ends
   (Colab storage is temporary).

### 5.3 Kaggle Notebook

1. Settings -> Accelerator -> **GPU P100**; Settings -> Internet -> **On**.
   Both need a phone-verified Kaggle account.
2. Run the same cells as Colab, with `SYSTEM_LABEL = "Kaggle-P100"`.
3. Download the CSV from the **Output** panel (files under `/kaggle/working/`).

Always run a 20-sample test (`MAX_SAMPLES = 20`) first and check the GPU name in
the diagnostics block at the top of the output matches the label you set.

---

## 6. Results

All numbers are from the three files in `results/` (750 samples each).

### 6.1 Accuracy (identical across hardware for NLI and Retrieval)

| Detector | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| NLI-based | 0.50 | 0.731 | 0.464 | 0.568 |
| Retrieval-based | 0.55 | 1.000 | 0.004 | 0.008 |
| Retrieval-based | 0.65 | 0.619 | 0.025 | 0.049 |
| Retrieval-based | 0.75 | 0.739 | 0.133 | 0.226 |
| Retrieval-based | 0.85 | 0.720 | 0.499 | 0.590 |
| Retrieval-based | 0.90 | 0.709 | 0.730 | 0.719 |
| SelfCheck-style (RTX3050, CUDA) | 0.50 | 0.719 | 0.816 | 0.764 |
| SelfCheck-style (RTX3050, CPU) | 0.50 | 0.723 | 0.800 | 0.760 |
| SelfCheck-style (Colab-T4) | 0.50 | 0.716 | 0.789 | 0.750 |
| SelfCheck-style (Kaggle-P100) | 0.50 | 0.717 | 0.767 | 0.741 |

- NLI and Retrieval give **bit-identical predictions** on all three systems and
  on both CPU and GPU. Hardware changes only speed and memory, not accuracy.
- SelfCheck's F1 varies between 0.741 and 0.764 across runs. That spread comes
  from random sampling (no seed is set), not from the hardware.

### 6.2 Latency and peak GPU memory

| System / device | NLI ms | Retrieval@0.90 ms | SelfCheck ms | NLI MB | SelfCheck MB |
|---|---|---|---|---|---|
| RTX3050 / CPU | 1961.8 | 54.1 | 40168.5 | n/a | n/a |
| RTX3050 / CUDA | 70.7 | 21.4 | 5416.9 | 1657.9 | 2902.6 |
| Colab-T4 / CUDA | 72.6 | 15.2 | 2594.0 | 1658.9 | 2903.6 |
| Kaggle-P100 / CUDA | 78.8 | 14.0 | 2009.1 | 1658.9 | 2938.6 |

Observations:

- CPU -> GPU speedup on the RTX 3050: NLI **27.8x**, SelfCheck **7.4x**,
  Retrieval **2.5x**.
- Retrieval is the cheapest method by a wide margin (roughly 14-21 ms, a few
  hundred MB at most). SelfCheck is 100x-400x slower than Retrieval and needs
  about 2.9 GB of GPU memory.
- NLI latency is nearly the same on all three GPUs (71-79 ms). With batch size 1
  the GPU is probably not the bottleneck there.
- SelfCheck is about 2-2.7x slower on the RTX 3050 than on the cloud GPUs, but see
  limitation 2 below before drawing a conclusion from that.

### 6.3 Chance-corrected metrics (derived)

Derived from the results files using 750 samples / 511 positives (all implied
confusion-matrix counts came out as whole numbers, which supports those totals).

| Detector (best config) | F1 | Balanced accuracy | MCC |
|---|---|---|---|
| NLI-based | 0.568 | 0.550 | 0.094 |
| Retrieval-based @ 0.90 | 0.719 | 0.545 | 0.091 |
| SelfCheck-style (RTX3050 / Colab / Kaggle) | 0.764 / 0.750 / 0.741 | 0.567 / 0.560 / 0.559 | 0.149 / 0.128 / 0.124 |
| Trivial baseline: flag everything | **0.810** | 0.500 | 0.000 |

**Read this before quoting any F1.** 68% of the samples are hallucinated, so a
detector that flags every summary scores F1 = 0.81, which is higher than all
three methods. Balanced accuracy of 0.55-0.57 and MCC of 0.09-0.15 mean all three
detectors are only slightly better than chance on this data. That is in line with
the FaithBench authors' own finding that existing detectors struggle on modern-LLM
hallucinations. A paper built on this data should lead with balanced accuracy /
MCC or report the baseline next to F1, not present F1 alone.

---

## 7. Limitations and open items

1. **Retrieval sweep hit its upper edge.** F1 rises at every step up to 0.90, the
   largest threshold tested. The true optimum may be higher; extend the sweep
   (0.95, 0.97, ...) before calling 0.90 "best".
2. **Code versions are not fully identical across systems.** The `RTX3050` run
   included CPU passes and may predate the `num_beams=1` patch, which Colab and
   Kaggle used. If so, its SelfCheck latency (5.4 s) is not comparable to
   theirs. **Confirm, or rerun `RTX3050` SelfCheck on GPU with the patched script.**
   Also commit the final patched script to the repo; the committed version may
   still be the unpatched one, because the patches were applied with `sed` inside
   the Colab/Kaggle sessions.
3. **Thresholds were tuned and evaluated on the same data.** There is no
   held-out split, so the Retrieval@0.90 result is optimistic. The NLI and
   SelfCheck thresholds (0.5) were not tuned.
4. **Peak-memory numbers for Retrieval are not trustworthy.** They jump around
   (for example 100.7 -> 187.3 -> 274 -> 100.7 MB on Colab) for the same tiny
   model, which points to leftover allocations from earlier runs rather than a
   real threshold effect. NLI (about 1.66 GB) and SelfCheck (about 2.9 GB) are
   stable across systems and are the figures to quote.
5. **SelfCheck is stochastic and unseeded.** Fix the random seed and, ideally,
   repeat runs to report a mean and spread.
6. **Truncation by characters.** Sources are cut to 1024 characters for NLI, which
   may drop the part of the source needed to verify a claim and depress recall.
7. **Latency** is per-sample at batch size 1 and includes tokenization. It is
   not a throughput benchmark. Results come from one run per system, with no
   confidence intervals.
8. **Single dataset, single task** (summarization hallucination on FaithBench).
9. **Hardware details to confirm:** earlier notes called the local GPU a 4 GB
   card, but `nvidia-smi` reported 6144 MiB. Check the "GPU total memory" line
   printed by the script and use that number consistently.

---

## 8. Troubleshooting log (what went wrong and how it was fixed)

| Symptom | Cause | Fix |
|---|---|---|
| Precision / recall / F1 all exactly 0.0 | The labelling script matched `unwanted.intrinsic`, but FaithBench spells it `Instrinsic`, so no sample was ever labelled hallucinated | Match case-insensitively on the substring `unwanted` / `questionable` (`02_prepare_dataset_FIXED.py`) |
| `No batch files found ... scripts\data\raw` | The patched script dropped the `..` and looked inside `scripts/` | Paths restored to `../data/raw` |
| `PermissionError` writing `faithbench_clean.csv` on Windows | File locked by another program (e.g. open in Excel) | Close it or delete the file, then rerun |
| GPU memory column 0.0 / everything on CPU | CPU-only PyTorch wheel installed | Reinstall the CUDA build (`cu130` for driver 591.66) |
| `pip install torch --index-url .../cu121` fails | That CUDA wheel index no longer carries current PyTorch releases | Use the selector on pytorch.org to find the current index |
| Retrieval detector scores 0 at threshold 0.55 | A fixed threshold of 0.55 almost never fires; source-summary similarity is high even for hallucinated text | Sweep thresholds 0.55-0.90 |
| SelfCheck stuck at 0/750 on Colab | Default generation settings were slow (beam search) | `num_beams=1` in `generate()` |
| Warning: both `max_new_tokens` and `max_length` set | Model's default config conflicts with our setting | Harmless; `max_new_tokens` wins |
| Kaggle: `No such file or directory` for the script | Wrong script name and wrong working directory | Use `03_run_detectors_v3.py` and `cd` into the repo first |
| Kaggle GPU option greyed out | Account not phone-verified, or weekly quota used | Verify phone number; check quota |

---

## 9. Project history, in order

1. Chose the topic: hallucination detection for LLM summaries, with a hardware-
   cost angle that existing papers (FactCC, SummaC, SelfCheckGPT, FRANK,
   FaithBench) do not cover.
2. Picked FaithBench as the dataset because it contains hallucinations from
   modern LLMs and needs no labelling by us.
3. Built the pipeline: download, clean, run detectors, record accuracy plus
   latency plus memory.
4. Fixed the labelling bug, path bug, and CUDA install (section 8).
5. Added the third detector (SelfCheck-style), the retrieval threshold sweep, the
   `SYSTEM_LABEL` tag, and a CPU/GPU device loop.
6. Ran the full 750-sample benchmark on the RTX 3050 (CPU and GPU).
7. Ran GPU-only benchmarks on Colab (T4) and Kaggle (P100). The AWS plan was
   dropped for lack of credits.
8. Analysed the three result files (section 6) and wrote this README.

---

## 10. Cite / credit

- Bao et al., 2025. *FaithBench: A Diverse Hallucination Benchmark for
  Summarization by Modern LLMs.* NAACL 2025.
- Kryscinski et al., 2020. *Evaluating the Factual Consistency of Abstractive
  Text Summarization.* EMNLP 2020.
- Laban et al., 2022. *SummaC: Re-Visiting NLI-based Models for Inconsistency
  Detection in Summarization.* TACL.
- Manakul et al., 2023. *SelfCheckGPT: Zero-Resource Black-Box Hallucination
  Detection for Generative Large Language Models.* EMNLP 2023.
- Pagnoni et al., 2021. *Understanding Factuality in Abstractive Summarization
  with FRANK.* NAACL 2021.

Verify each citation against the publisher page before using it in a paper.
FaithBench data remains subject to its own license; check the FaithBench repo.

## License

_To be added._
