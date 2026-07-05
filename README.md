# Deep Learning Systems (CNRPark Occupancy CNN)
![](https://github.com/user-attachments/assets/caff4e70-ee8d-41a1-b789-e30d345aed08)

A controlled deep-learning experiment: two PyTorch CNNs for binary parking-space
occupancy classification, compared under a **single-change** design. The baseline
is a 3-block CNN; the experimental model adds **one extra convolution block**
(and nothing else), so any performance difference is attributable to that one
architectural lever.

This is Project 4 of the Udacity AI Mastery capstone.

---

## What the project does

- **Task:** binary image classification — predict `busy` (occupied) vs. `free`
  (empty) for a 150×150 RGB parking-lot patch.
- **Dataset:** CNRPark (Amato et al., ISTI-CNR, 2016) — 12,584 real camera
  patches from two cameras (A, B), labeled by folder. Public, non-synthetic, and
  not reused from any prior capstone project. Home: <http://cnrpark.it/>
- **Models:** a 3-block `BaselineCNN` (5.4M params) and a 4-block
  `ExperimentalCNN` (3.0M params) that differ by exactly one conv block. Same
  `Flatten → FC` head, same loss/optimizer/split/augmentation.
- **Evaluation:** single-seed 20-epoch run plus a 3-seed (7, 42, 99) paired
  comparison, with a per-camera/per-class breakdown to expose domain shift.
- **Headline finding:** on this dataset the deeper model did **not** help — the
  baseline wins the multi-seed mean (−0.42 pp for the deeper model, losing 3/3
  seeds) despite the deeper model using 1.8× fewer parameters. The deeper model
  also shows late-epoch training instability and a camera-B weakness the
  baseline does not. Full numbers and interpretation live in the report.

---

## Repository layout

```
project_04_DeepLearning/
├── data/                          # NOT in version control (see .gitignore)
│   ├── CNRPark-Patches-150x150/   # the patches: <camera>/<busy|free>/*.jpg
│   ├── CNRPark+EXT.csv            # per-frame metadata (follow-up ablations)
│   └── CNRPark-Patches-150x150.zip
├── notebooks/
│   └── deep_learning.ipynb        # the rubric notebook: load, train, compare, summarize
├── src/
│   ├── dataset.py                 # patch scanning, transforms, DataLoader split (SEED=42)
│   ├── models.py                  # BaselineCNN + ExperimentalCNN + get_model() factory
│   └── train.py                   # fit() loop, per-epoch metrics, CSV history, CLI entry
├── reports/
│   ├── Deep_Learning_Systems_Analysis_Report.md   # the analysis report as MD file
│   ├── Deep_Learning_Systems_Analysis_Report.pdf   # the analysis report PDF file
│   ├── metrics_baseline.csv       # per-epoch history from the baseline run
│   └── metrics_experimental.csv   # per-epoch history from the experimental run
├── requirements.txt               # pip freeze of the project venv
├── .gitignore                     # ignores cnn_env/, __pycache__/, and data/
└── README.md                      # this file
```

### Source modules

- **`src/dataset.py`** — `scan_patches()` walks `<camera>/<busy|free>/*.jpg` and
  tags labels (busy→1, free→0). `get_transforms(train)` applies
  `Resize(150) → ToTensor → Normalize(ImageNet)` with random flips added for
  training only. `build_dataloaders()` carves a SEED=42 train/val/test split
  (80/10/10) so both models see identical data.
- **`src/models.py`** — `BaselineCNN` (3 conv blocks, 32→64→128, then
  `Flatten → Linear(128·18·18, 128) → Linear(128, 1)`) and `ExperimentalCNN`
  (4 conv blocks, 32→64→128→256, then `Flatten → Linear(256·9·9, 128) →
  Linear(128, 1)`). `get_model(name)` is the factory used by `train.py`. A
  `__main__` block runs a forward-pass shape check on both models.
- **`src/train.py`** — `run_epoch()` does one forward/backward pass;
  `fit()` trains one model for N epochs with `BCEWithLogitsLoss` + `Adam(lr=1e-3)`
  and returns per-epoch history plus the final test score; `save_csv()` writes
  the history to `reports/metrics_<name>.csv`. `DEVICE = cuda if available else
  cpu`. `main()` is the CLI entry point.

---

## Dataset access

The dataset is **not** committed (it is large and licensed for research /
educational use). To reproduce, download CNRPark from
<http://cnrpark.it/> and unzip so the layout matches:

```
data/CNRPark-Patches-150x150/
    A/busy/*.jpg
    A/free/*.jpg
    B/busy/*.jpg
    B/free/*.jpg
```

The optional `data/CNRPark+EXT.csv` (per-frame metadata: camera, datetime,
weather, occupancy, slot_id) supports follow-up weather-stratified ablations but
is not needed for the baseline-vs-experiment comparison.

---

## How to run

All commands assume the working directory is `project_04_DeepLearning/`.

### 1. Create / activate the environment

```bash
python -m venv cnn_env
# Windows
cnn_env\Scripts\activate
# macOS / Linux
source cnn_env/bin/activate

pip install -r requirements.txt
```

`requirements.txt` is a `pip freeze` of the environment the project was built
in (`torch==2.12.1+cu126`, `torchvision==0.27.1+cu126`, plus pandas, matplotlib,
seaborn, jupyterlab, nbformat, nbconvert). For the CUDA build, use the PyTorch
wheel index that matches your GPU/driver; a CPU-only torch also works but is
slower.

### 2. Smoke-test the model shapes (no training, no data needed)

```bash
python src/models.py
# expected:
#   baseline: OK -> (2, 1)
#   experimental: OK -> (2, 1)
```

### 3. Reproduce the headline numbers (CLI, no notebook required)

```bash
python src/train.py \
    --patches-dir data/CNRPark-Patches-150x150 \
    --epochs 20 \
    --batch-size 64 \
    --out-dir reports
```

This trains both models on the same split and writes
`reports/metrics_baseline.csv` and `reports/metrics_experimental.csv`, then
prints final `test_acc` / `test_loss` / `best_val_acc` for each.

### 4. Run the notebook (full experiment + plots + comparison)

```bash
jupyter lab
```

Open `notebooks/deep_learning.ipynb` and choose **Kernel → Restart & Run All**.
The notebook loads the data, trains both models, plots loss/accuracy curves,
builds confusion matrices, runs the 3-seed paired comparison, and produces the
per-camera/per-class breakdown. Expected runtime is roughly 8 minutes for the
20-epoch run plus 5 minutes for the 3-seed rerun on an NVIDIA GTX 1650 Ti.

---

## Reproducibility

- A single `SEED = 42` is used in `src/dataset.py` (the data split) and
  `src/train.py` (default model init + shuffle seed). Both models see the same
  split indices, so the only difference between runs is the model.
- Multi-seed runs vary only the model's `seed=` argument; the data split is held
  constant, so each (baseline, experimental) pair shares a per-seed noise floor
  and the paired difference is the test signal.
- Per-epoch metrics are persisted to `reports/metrics_*.csv` so a Restart & Run
  All reproduces the reported curves.
- The full environment is captured in `requirements.txt` (a `pip freeze`).

---

## Results summary

| model | params | test_acc (20-epoch, SEED=42) | multi-seed mean test_acc (3 seeds) |
|---|---|---|---|
| BaselineCNN (3 blocks) | 5,401,921 | 0.9976 | 0.9923 |
| ExperimentalCNN (4 blocks) | 3,042,881 | 0.9849 | 0.9881 |

The baseline wins the single-seed run (+1.27 pp) and the multi-seed mean
(+0.42 pp, winning all 3 individual seeds). The deeper model's `best_val_acc`
ties the baseline (0.9968), so it *can* reach baseline-level performance — its
final-epoch test score is dragged down by a late-training val_loss collapse and a
camera-B weakness the baseline does not share. Interpretation, limitations, and
follow-ups are in the analysis report.

---

## License and ethics

CNRPark is released by ISTI-CNR for research and educational use. The 150×150
patches contain no biometric data (no faces, no readable license plates at this
resolution) and no PII beyond approximate timestamps and weather already in the
dataset metadata. The trained models are not deployed in any real surveillance
system; a production deployment would need its own camera-specific bias
evaluation and consent framework. The full ethics discussion is in the report.