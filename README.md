# VER: Discrimination of Persons and Objects in the Ultrasonic Beam

Machine learning pipeline for classifying ultrasonic sensor readings from a **vertically oriented sensor** into three classes: **floor** (0), **box** (1), **human** (2). The project follows the VER topic from the course (FIUS sensor, pulse-echo method, FFT → features → classifier).

---

## Repository structure

```
mlprojCursor/
├── VER_pipeline.ipynb      # Full pipeline: merge → load → visualize → train → evaluate
├── Dataset/                # Raw and derived data
│   ├── floor_combined.csv  # Class 0
│   ├── box_combined.csv    # Class 1
│   ├── human_combined.csv  # Class 2
│   ├── merged_dataset.csv  # Single merged CSV (created by notebook or merge script)
│   └── merged_small.csv    # Subset for fast runs (optional)
├── results/                # Confusion matrices, metrics, plots
├── models/                 # Saved scaler and models (from train_all_models.py)
├── Project_Docs/            # Project's guideline (PDF, DOCX, XLSX)
├── merge_datasets.py       # CLI: merge the three CSVs into one
├── create_small_merged.py  # CLI: build a smaller CSV for memory-limited runs
├── train_all_models.py    # CLI: train SVM, RF, CNN, TCN, Transformers; save results
├── requirements.txt
└── README.md
```

---

## Setup

- **Python**: 3.8+
- **Dependencies**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
pip install -r requirements.txt
```

For the optional deep-learning models (CNN, TCN, Transformers) run via `train_all_models.py`:

```bash
pip install torch
```

---

## Data

- **Source files**: `Dataset/floor_combined.csv`, `Dataset/box_combined.csv`, `Dataset/human_combined.csv`. Each row is one recording; **label is in column index 2** (0=floor, 1=box, 2=human); remaining columns are features (e.g. FFT/signal).
- **Merged file**: The notebook (or `merge_datasets.py`) produces `Dataset/merged_dataset.csv` by concatenating the three files and ensuring the label is set in column 2.
- **Small subset**: For large merged files (~25k columns), `merged_small.csv` (fewer rows, every Nth feature) can be generated for faster, memory-friendly runs.

---

## How to run

### Recommended: single notebook

Open and run **`VER_pipeline.ipynb`** top to bottom. The notebook **does not merge** datasets; use **`Dataset/merged_dataset.csv`** (or create it once with **`python merge_datasets.py -o Dataset/merged_dataset.csv`**). The notebook then:

1. **Load** — Uses `merged_dataset.csv` or, if present, `merged_small.csv` (subset for fast runs).
2. **Display** — Shape, class counts, and a short summary table.
3. **Visualize** — Class distribution bar chart; distribution of feature means; mean signal (first N features) per class.
4. **Preprocess** — Stratified train/test split, `StandardScaler`.
5. **Train all models** — SVM, Random Forest, and (if PyTorch is installed) CNN, TCN, Conv-Transformer Hybrid, Time Series Transformer.
6. **Evaluate** — For each model: confusion matrix, classification report, pairwise discriminant difference ΔD.
7. **Compare** — Summary table (accuracy, macro F1) and bar chart; best model by accuracy and by macro F1.

Figures are saved under `results/` (e.g. `data_overview.png`, `confusion_all_models.png`, `comparison_all_models.png`). Install PyTorch (`pip install torch`) to run the four deep learning models in the same notebook.

### Optional: command-line scripts

- **Merge only**  
  `python merge_datasets.py -o Dataset/merged_dataset.csv`  
  Use `--max-rows-per-class N` to cap rows per class.

- **Create small merged file**  
  `python create_small_merged.py --max-rows 5000 --feature-step 50`

- **Train all models (SVM, RF, CNN, TCN, Transformers)** and save metrics/plots:  
  `python train_all_models.py --data Dataset/merged_small.csv --feature-step 1 --n-cols 504 --max-samples 4000`

---

## Models and metrics

| Model | Description |
|-------|-------------|
| SVM | RBF kernel, C=1, gamma=scale |
| Random Forest | 100 trees, max_depth=20 |
| CNN | 1D conv (notebook runs SVM/RF only; full set via `train_all_models.py`) |
| TCN | Temporal convolutional network |
| Conv-Transformer | 1D conv + Transformer encoder |
| Time Series Transformer | Patch embedding + Transformer encoder |

**Evaluation (aligned with course material):**

- **Confusion matrix** — Rows = true class, columns = predicted; TP, FP, FN, TN per class.
- **Classification report** — Precision, recall, F1 (per class and macro/weighted).
- **Discriminant difference** — ΔD = (D₁ − D₂)² / (D₁ + D₂)² for pairs of classes (using diagonal counts).

---

## Errors and troubleshooting

- **FileNotFoundError for merged/small file** — Run the notebook from the project root so `Dataset/` is found; run the “Merge” and “Create small dataset” cells first.
- **MemoryError when loading full merged file** — Use `merged_small.csv` or create it with `create_small_merged.py` (fewer rows/columns).
- **Only one class in training** — Ensure the merged (or small) file contains all three labels; the notebook’s small-dataset step samples randomly so all classes appear.

---

## References

- **Project_Docs/** — Topic description, process steps, deliverables, and KPIs.
- Sensor pipeline (from course material): Sonic wave → Reception → DAQ → FFT → Feature extraction → Decision.
