# Active-Tactile-Perception-for-Fine-Grained-Classification-of-Visually-Indistinguishable-Liquids

## Overview

This project implements a Graph Neural Network (GNN) + Transformer fusion architecture for classifying multi-channel electronic signals. The pipeline processes raw CSV time-series data through STFT transformation, trains a spatial-temporal model, and provides comprehensive evaluation and diagnostic tools.

### Problem Statement
- **Input**: Multi-channel CSV files containing time-series signals (10 active channels out of 12)
- **Output**: Classification into N classes (configurable, default 6)
- **Key Features**: 
  - Signal preprocessing with trigger-based segmentation
  - STFT frequency-domain feature extraction
  - Graph-based spatial modeling + Transformer temporal modeling
  - Comprehensive evaluation metrics and deep diagnostics

---

## Environment Setup

### Requirements
- **Python**: 3.7+ (tested on 3.8-3.10)
- **CUDA**: Optional but recommended (PyTorch will auto-detect)

### Installation

```bash
# Clone/download the project
cd gnn_transformer_complete

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version
pip install numpy pandas scikit-learn tqdm matplotlib seaborn umap-learn
```

**Core Dependencies:**
- `torch` (>=1.10)
- `numpy`, `pandas`
- `scikit-learn` (for train/test split, metrics)
- `scipy` (for STFT)
- `tqdm` (progress bars)
- `umap-learn` (for Step6 diagnostics)

---

## Project Structure

```
configs/config.py      # Central configuration file (ALL parameters here)
scripts/
  step1_csv_trim.py    # CSV signal trimming
  step2_csv_reorder_align.py  # Data consolidation & balancing
  step3_stft.py        # STFT transformation
  step4_check.py       # Data dimension validation
  step5_train.py       # Model training & evaluation
  step6_diagnostic.py  # Deep model diagnostics
src/
  data/                # Dataset loaders
  models/              # GNN-Transformer architecture
  utils/               # Evaluation & experiment tracking
  diagnostics/         # Diagnostic modules
```

---

## How to Run

### **Pre-Flight: Configure Paths**

**Edit `configs/config.py` before running any script:**

```python
# Key paths to update:
RAW_CSV_DIR = "Data/2025-12-18/csv_logs"         # Your raw CSV folder
STEP3_INPUT_DIR = "input_data/processed_main/sauce-like"  # After Step2
DATA_ROOT = "input_data/processed_main/sauce-like_Class15_STFT"  # After Step3
```

---

### **Step 1: CSV Signal Trimming**

**Purpose**: Extract main segments and sub-segments from raw CSV files using trigger thresholds.

**Command:**
```bash
python scripts/step1_csv_trim.py
```

**Input:**
- Raw CSV files in `RAW_CSV_DIR` (default: `Data/2025-12-18/`)
- Each CSV must contain columns: `time_ms`, `ch0`, `ch1`, ..., `ch15`

**Output:**
- `trimmed_data/processed_main/`: Main signal segments
  - Organized as: `[Liquid]_[Class]/[Liquid]_[Class]_XXXX.csv`
- `trimmed_data/processed_sub/high/`: High-level sub-segments
- `trimmed_data/processed_sub/low/`: Low-level sub-segments

**Key Parameters** (in `config.py`):
- `NAME_LIST`: Liquid names for cyclic naming (e.g., `['yogurt', 'milk', ...]`)
- `TRIGGER_CH`, `TRIGGER_THRESHOLD`: Main trigger channel (default: `ch10`, threshold `15000`)
- `SUBCUT_CH`, `SUBCUT_THRESHOLD`: Sub-segment trigger (default: `ch8`, threshold `15000`)
- `MIN_SEGMENT_DURATION_MS`: Minimum segment length (default: `2000ms`)

**Processing Logic:**
1. Scans all CSVs in `RAW_CSV_DIR`
2. Detects trigger crossings in `TRIGGER_CH`
3. Extracts segments exceeding minimum duration
4. Applies median filtering (kernel size: `MEDIAN_FILTER_KERNEL`)
5. Saves main segments to `processed_main`
6. Further splits main segments at `SUBCUT_CH` triggers → `processed_sub/high` and `processed_sub/low`

---

### **Step 2: Data Consolidation & Balancing**

**Purpose**: Merge multiple data sources, standardize naming, and balance class samples.

**Command:**
```bash
python scripts/step2_csv_reorder_align.py
```

**⚠️ Important**: This script uses **hardcoded paths at the top of the file**, NOT `config.py`.

**Configuration** (edit `step2_csv_reorder_align.py` lines 30-50):
```python
INPUT_ROOTS = [
    r"Data/2025-12-18/csv_logs",
    # Add more input directories here
]
OUTPUT_ROOT = r"Code/code_electronic_two_stream/input_data"
COPY_INSTEAD_OF_MOVE = True  # True=copy files, False=move files
BALANCE_MAIN_SAMPLES = True  # Enable class balancing
```

**Input:**
- Multiple folders containing `processed_main/` or `ProcessedMan/` subfolders
- And `processed_sub/high` (or `ProcessedSub/Hi`) and `processed_sub/low` (or `ProcessedSub/Lo`)

**Output:**
- `OUTPUT_ROOT/processed_main/[Liquid]_[Class]/`: Consolidated main segments
  - Renamed: `[Liquid]_[Class]_0001.csv`, `_0002.csv`, ...
  - If `BALANCE_MAIN_SAMPLES=True`: Excess samples moved to `extra/` subfolder
- `OUTPUT_ROOT/processed_sub/Hi/[Liquid]_[Class]/`: Consolidated high-level sub-segments
  - Named: `[Liquid]_[Class]_Hi_0001.csv`, ...
- `OUTPUT_ROOT/processed_sub/Lo/[Liquid]_[Class]/`: Consolidated low-level sub-segments
  - Named: `[Liquid]_[Class]_Lo_0001.csv`, ...

**Processing Logic:**
1. Scans all `INPUT_ROOTS` for compatible folder structures
2. Parses liquid/class names from filenames (e.g., `milk_Class5_xxx.csv` → liquid=`milk`, class=`Class5`)
3. Renames files with zero-padded counters (per liquid-class combination)
4. If balancing enabled: Finds global minimum sample count, moves excess to `extra/` folders

---

### **Step 3: STFT Transformation**

**Purpose**: Convert time-domain CSV signals to frequency-domain NPY arrays.

**Command:**
```bash
python scripts/step3_stft.py
```

**Input:**
- CSV files in `STEP3_INPUT_DIR` (configured in `config.py`)
- Default: `input_data/processed_main/sauce-like`

**Output:**
- NPY files in `STEP3_OUTPUT_DIR` (default: adds `_STFT` suffix)
- Example: `sauce-like_Class15_STFT/[Liquid]_[Class]/[Liquid]_[Class]_0001.npy`
- Each NPY shape: `(num_channels, 1, freq_bins, time_bins)` = `(12, 1, 101, 193)` by default

**Key Parameters** (in `config.py`):
```python
STFT_NPERSEG = 512       # STFT window size
STFT_NOVERLAP = 384      # Overlap samples
STFT_NFFT = 512          # FFT points
STFT_FMAX = 1000.0       # Maximum frequency (Hz)
STFT_SHAPE = (101, 193)  # Output (freq_bins, time_bins)
SAMPLING_RATE = 5000     # Sampling rate (Hz)

# Directory filtering
INCLUDE_DIR_KEYWORDS = ["Class15"]  # Only process folders matching these keywords
EXCLUDE_DIR_KEYWORDS = ['_STFT']   # Skip folders with these keywords
INCLUDE_KEYWORDS_MODE = 'OR'        # 'OR' or 'AND' for multiple keywords
KEYWORD_MATCH_MODE = 'name'         # 'name' or 'path'
```

**Processing Logic:**
1. Recursively scans `STEP3_INPUT_DIR` for matching subdirectories
2. For each CSV:
   - Reads 10 valid channels (excludes channels 4 and 9, which are missing)
   - Pads to 12 channels with zeros for missing channels
   - Applies STFT per channel → complex spectrogram
   - Takes magnitude, crops to `STFT_FMAX` frequency
   - Resizes to `STFT_SHAPE` using interpolation
   - Saves as NPY: shape `(12, 1, 101, 193)`

---

### **Step 4: Data Validation**

**Purpose**: Verify dataset output dimensions before training.

**Command:**
```bash
python scripts/step4_check.py
```

**Input:**
- STFT NPY files in `DATA_ROOT` (configured in `config.py`)

**Output:**
- Console printout showing:
  - Expected vs. actual STFT shape: `(12, 1, 101, 193)`
  - Expected vs. actual adjacency matrix shape: `(12, 12)`
  - Sample label

**Expected Output:**
```
✅ 维度验证通过，可正常训练
```

**Failure Modes:**
- `❌ 数据集为空`: Check `DATA_ROOT` path
- `❌ 维度不匹配`: Check `STFT_SHAPE`, `NUM_CHANNELS` in `config.py`

---

### **Step 5: Model Training & Evaluation**

**Purpose**: Train the GNN-Transformer model and evaluate on test set.

**Command:**
```bash
python scripts/step5_train.py
```

**Input:**
- STFT NPY files in `DATA_ROOT`

**Output:**
- **Checkpoints** in `checkpoints/[dataset_name]/`:
  - Filename format: `exp_[8-char-hash]_acc_0.XXXX.pth`
  - Contains: `model_state_dict`, `epoch`, `val_acc`
- **Evaluation Results** in `evaluation_results/[dataset_name]/`:
  - `config_snapshot.json`: Complete config at training time
  - `experiment_metadata.json`: Experiment ID, creation time, data paths
  - `hyperparameters.csv`: All training hyperparameters
  - `model_architecture.txt`: Model structure summary
  - `[exp_id]_eval_metrics.csv`: Accuracy, Precision, Recall, F1, MSE, RMSE
  - `[exp_id]_confusion_matrix.csv`: Per-class confusion matrix (normalized)
  - `[exp_id]_learning_curve.csv`: (if `EVAL_ENABLE_LEARNING_CURVE=True`)

**Key Hyperparameters** (in `config.py`):
```python
# Training
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VALID_RATIO = 0.2  # Test set ratio (if using stratified_ratio split)
RANDOM_SEED = 42

# Model architecture
NUM_CLASSES = 6             # Auto-detected from DATA_ROOT if mismatch
GNN_HIDDEN_DIM = 64
TRANSFORMER_NHEAD = 1
TRANSFORMER_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 256

# Data augmentation (applied during training)
ENABLE_NORMALIZE = True     # Z-score normalization
ENABLE_AUGMENT = True       # Random augmentation
AUG_PROB = 0.8              # Probability of applying augmentation
AUG_NOISE_STD = 0.02        # Gaussian noise std
AUG_GAIN_RANGE = (0.8, 1.2) # Random gain multiplier
AUG_TIME_SHIFT = 8          # Max time-axis shift (bins)
AUG_FREQ_MASK_MAX = 12      # Max frequency mask size
AUG_TIME_MASK_MAX = 18      # Max time mask size

# Data split strategy
SPLIT_STRATEGY = "stratified_ratio"  # Options: "random", "stratified_ratio", "stratified_fixed"
TEST_SAMPLES_PER_CLASS = 20          # Only for "stratified_fixed" mode
```

**Data Split Strategies:**
1. **`random`**: Completely random split (fast prototyping)
2. **`stratified_ratio`** (default): Maintains class distribution, splits by `VALID_RATIO`
3. **`stratified_fixed`**: Fixed test samples per class (e.g., 20/class for publications)

**Training Flow:**
1. Auto-detects number of classes from `DATA_ROOT` folder structure
2. Loads STFT dataset and splits train/test
3. Initializes GNN-Transformer model
4. Trains for `EPOCHS` epochs with AdamW optimizer
5. Saves best checkpoint (highest validation accuracy)
6. Loads best model and runs full evaluation:
   - Standard metrics (accuracy, precision, recall, F1)
   - Confusion matrix
   - Learning curve (if enabled, requires multiple training runs)

**Experiment Session Management:**
- `evaluation_results/.current_session.json` tracks active experiment
- If hyperparameters unchanged → reuses same folder
- If hyperparameters changed → creates new folder with incremented suffix (`_001`, `_002`, ...)

---

### **Step 6: Deep Model Diagnostics**

**Purpose**: Extract embeddings, analyze separability, and perform sensitivity tests.

**Command:**
```bash
python scripts/step6_diagnostic.py
```

**Interactive Workflow:**
1. Prompts for checkpoint path (can be folder or `.pth` file)
2. Matches checkpoint to existing evaluation folder or creates new one
3. Asks to overwrite if diagnostics already exist
4. Runs 10 diagnostic modules (see below)

**Input:**
- Trained checkpoint (`.pth` file from Step5)
- Test dataset (same as Step5)

**Output** (in `evaluation_results/[dataset_name]/diagnostics/`):
1. **Embedding Export**:
   - `diag_raw_stft_embeddings.csv`: Raw STFT features
   - `diag_node_feat_embeddings.csv`: After GNN layer
   - `diag_transformer_embeddings.csv`: After Transformer layer
   - `diag_metadata.csv`: Sample IDs, labels, split info
2. **Dimensionality Reduction**:
   - `diag_pca_2d_[layer].csv`: PCA 2D projections
   - `diag_pca_3d_[layer].csv`: PCA 3D projections
   - `diag_umap_2d_[layer].csv`: UMAP 2D projections
3. **Separability Metrics**:
   - `diag_separability_metrics.csv`: Linear classifier accuracy per layer
4. **Misclassification Analysis**:
   - `diag_misclassification_analysis.csv`: Error patterns by class and confidence
5. **Frequency Sensitivity**:
   - `diag_frequency_sensitivity.csv`: Accuracy drop when masking frequency bands
6. **Time Sensitivity**:
   - `diag_time_sensitivity.csv`: Accuracy drop when masking time windows
7. **Auto-Scan Results**:
   - `diag_auto_scan_results.csv`: Top-K most sensitive frequency/time regions
8. **Channel Ablation**:
   - `diag_channel_ablation.csv`: Accuracy when zeroing each channel

**Diagnostic Configuration** (in `config.py`):
```python
DIAG_ENABLE = True  # Master switch
DIAG_ENABLE_EMBEDDING_EXPORT = True
DIAG_EXPORT_LAYERS = ['raw_stft', 'node_feat', 'transformer']
DIAG_ENABLE_DIMENSIONALITY_REDUCTION = True
DIAG_DIM_REDUCTION_METHODS = ['pca_2d', 'pca_3d', 'umap_2d']
DIAG_ENABLE_SEPARABILITY_METRICS = True
DIAG_ENABLE_MISCLASSIFICATION_ANALYSIS = True
DIAG_ENABLE_FREQUENCY_SENSITIVITY = True
DIAG_FREQ_MASK_BANDS = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 2000)]
DIAG_ENABLE_TIME_SENSITIVITY = True
DIAG_TIME_MASK_WINDOWS = 10
DIAG_ENABLE_AUTO_SCAN = True
DIAG_ENABLE_CHANNEL_ABLATION = True
DIAG_ABLATION_MODE = 'zero'  # 'zero' or 'drop'
```

---

## Input Data Requirements

### CSV File Format (Step1 Input)
- **Required columns**:
  - `time_ms`: Timestamp in milliseconds
  - `ch0`, `ch1`, ..., `ch15`: 16 channels of signal data (float)
- **Used channels**: `ch0`, `ch1`, `ch2`, `ch3`, `ch6`, `ch7`, `ch8`, `ch10`, `ch14`, `ch15` (10 channels)
- **Missing channels**: 4, 9 (padded with zeros in STFT)
- **Naming convention**: No strict requirement (Step1 renames using `NAME_LIST`)

### Folder Structure (Step2 Input)
- Each input root must contain:
  - `processed_main/` or `ProcessedMan/`: Main segments
  - `processed_sub/high/` (or `ProcessedSub/Hi/`): High-level sub-segments
  - `processed_sub/low/` (or `ProcessedSub/Lo/`): Low-level sub-segments
- CSV filenames should contain liquid and class info (e.g., `milk_Class5_*.csv`)

### NPY Array Format (Step3 Output, Step4-6 Input)
- **Shape**: `(12, 1, 101, 193)` (channels, dummy_dim, freq_bins, time_bins)
- **Data type**: `float32`
- **Normalization**: Applied per-sample during training if `ENABLE_NORMALIZE=True`

---

## Configuration Management

**All parameters are centralized in `configs/config.py`.**

### Critical Path Variables:
```python
PROJECT_ROOT         # Auto-detected (parent of configs/)
RAW_CSV_DIR          # Step1 input
STEP1_OUTPUT_MAIN    # Step1 output (main segments)
STEP1_OUTPUT_SUB     # Step1 output (sub segments)
STEP3_INPUT_DIR      # Step3 input (CSV folder to process)
STEP3_OUTPUT_DIR     # Step3 output (NPY folder)
DATA_ROOT            # Step4-6 input (STFT NPY folder)
CHECKPOINT_DIR       # Step5 checkpoint save path
EVAL_OUTPUT_DIR      # Step5-6 evaluation results
```

### How to Change Dataset:
1. Update `STEP3_INPUT_DIR` to point to new CSV folder
2. Update `DATA_ROOT` to new STFT output folder
3. Update `INCLUDE_DIR_KEYWORDS` to filter desired classes (e.g., `["Class10", "Class15"]`)
4. Update `NUM_CLASSES` if needed (auto-detected in Step5)

### How to Change Model:
- `GNN_HIDDEN_DIM`: GNN layer size
- `TRANSFORMER_LAYERS`: Number of Transformer encoder layers
- `TRANSFORMER_NHEAD`: Number of attention heads (must divide embedding dim)
- `TRANSFORMER_DIM_FEEDFORWARD`: Feedforward network size

---

## Reproducibility

### Random Seed Control
Set `RANDOM_SEED = 42` in `config.py` to ensure:
- Deterministic train/test split
- Reproducible model initialization
- Consistent augmentation (when `AUG_PROB < 1.0`)

### Deterministic Training (CUDA)
Step5 enables:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
**Trade-off**: Slower training but fully reproducible results.

### Experiment Tracking
- Each training run generates unique experiment ID (8-char hash of hyperparameters)
- `experiment_metadata.json` records:
  - Experiment ID
  - Creation timestamp
  - Data root path
  - Config snapshot
- Use `ExperimentTracker` to trace model → hyperparameters → data version

---

## Common Errors & Troubleshooting

### 1. `FileNotFoundError: DATA_ROOT does not exist`
**Cause**: `DATA_ROOT` in `config.py` points to non-existent folder.  
**Fix**: Run Step3 first, or update `DATA_ROOT` to existing STFT folder.

### 2. `RuntimeError: Expected shape (12, 1, 101, 193), got (...)`
**Cause**: STFT shape mismatch.  
**Fix**: Re-run Step3 with correct `STFT_SHAPE` in `config.py`, or update `STFT_SHAPE` to match existing NPY files.

### 3. `ValueError: Number of classes mismatch`
**Cause**: `config.NUM_CLASSES` doesn't match actual class folders in `DATA_ROOT`.  
**Fix**: Step5 auto-detects and updates `NUM_CLASSES`. Ensure class folder naming is consistent.

### 4. `ImportError: No module named 'umap'`
**Cause**: Missing UMAP dependency (required for Step6).  
**Fix**: `pip install umap-learn`

### 5. Step2 doesn't use `config.py` paths
**Cause**: Step2 has hardcoded configuration at file top.  
**Fix**: Edit `INPUT_ROOTS` and `OUTPUT_ROOT` directly in `step2_csv_reorder_align.py` (lines 30-45).

### 6. Step6 asks for checkpoint but folder not found
**Cause**: Checkpoint and evaluation folders have mismatched names.  
**Fix**: Use interactive selection (Step6 will list available folders) or manually match checkpoint folder name to evaluation folder.

### 7. CUDA out of memory
**Cause**: Batch size too large.  
**Fix**: Reduce `BATCH_SIZE` in `config.py` (try 4 or 2).

### 8. Training stuck at 0% accuracy
**Cause**: Learning rate too high or data not normalized.  
**Fix**: Enable `ENABLE_NORMALIZE=True` or reduce `LEARNING_RATE` to `1e-5`.

---

## Pipeline Summary (Quick Reference)

| Step | Script | Input | Output | Config Key |
|------|--------|-------|--------|------------|
| 1 | `step1_csv_trim.py` | Raw CSV | `trimmed_data/` | `RAW_CSV_DIR` |
| 2 | `step2_csv_reorder_align.py` | Multiple processed folders | `input_data/` | Hardcoded in script |
| 3 | `step3_stft.py` | CSV (from Step2) | NPY arrays | `STEP3_INPUT_DIR`, `STEP3_OUTPUT_DIR` |
| 4 | `step4_check.py` | NPY arrays | Console validation | `DATA_ROOT` |
| 5 | `step5_train.py` | NPY arrays | Checkpoint + evaluation | `DATA_ROOT`, `CHECKPOINT_DIR`, `EVAL_OUTPUT_DIR` |
| 6 | `step6_diagnostic.py` | Checkpoint + NPY | Diagnostic CSVs | `EVAL_OUTPUT_DIR` (auto-matches) |

**Minimal Workflow:**
```bash
# 1. Edit config.py paths
# 2. Run steps sequentially:
python scripts/step1_csv_trim.py
python scripts/step2_csv_reorder_align.py  # Edit hardcoded paths first!
python scripts/step3_stft.py
python scripts/step4_check.py  # Optional validation
python scripts/step5_train.py
python scripts/step6_diagnostic.py  # Paste checkpoint path when prompted
```

---

## Post-Run Folder Structure

```
gnn_transformer_complete/
├── configs/
│   ├── __init__.py
│   └── config.py
├── scripts/
│   ├── __init__.py
│   ├── step1_csv_trim.py
│   ├── step2_csv_reorder_align.py
│   ├── step3_stft.py
│   ├── step4_check.py
│   ├── step5_train.py
│   └── step6_diagnostic.py
├── src/
│   ├── data/
│   │   ├── csv_trimmer.py
│   │   ├── stft_processor.py
│   │   └── dataset.py
│   ├── models/
│   │   └── gnn_transformer.py
│   ├── utils/
│   │   ├── evaluator.py
│   │   └── experiment_tracker.py
│   └── diagnostics/
│       └── diagnostic_pipeline.py
├── Data/
│   └── 2025-12-18/
│       └── csv_logs/
│           ├── processed_main/
│           └── processed_sub/
├── trimmed_data/                    # Step1 output
│   ├── processed_main/
│   │   └── [Liquid]_[Class]/
│   │       └── [Liquid]_[Class]_XXXX.csv
│   └── processed_sub/
│       ├── high/
│       └── low/
├── input_data/                      # Step2 output
│   └── processed_main/
│       ├── [Liquid]_[Class]/
│       │   ├── [Liquid]_[Class]_0001.csv
│       │   └── extra/
│       └── sauce-like_Class15_STFT/ # Step3 output
│           └── [Liquid]_[Class]/
│               └── *.npy
├── checkpoints/                     # Step5 output
│   └── sauce-like_Class15_STFT_001/
│       └── exp_[hash]_acc_0.XXXX.pth
├── evaluation_results/              # Step5-6 output
│   ├── .current_session.json
│   └── sauce-like_Class15_STFT_001/
│       ├── config_snapshot.json
│       ├── experiment_metadata.json
│       ├── hyperparameters.csv
│       ├── model_architecture.txt
│       ├── [exp_id]_eval_metrics.csv
│       ├── [exp_id]_confusion_matrix.csv
│       └── diagnostics/
│           ├── diag_raw_stft_embeddings.csv
│           ├── diag_node_feat_embeddings.csv
│           ├── diag_transformer_embeddings.csv
│           ├── diag_metadata.csv
│           ├── diag_pca_2d_*.csv
│           ├── diag_separability_metrics.csv
│           ├── diag_misclassification_analysis.csv
│           ├── diag_frequency_sensitivity.csv
│           ├── diag_time_sensitivity.csv
│           └── diag_channel_ablation.csv
└── logs/
    └── sauce-like_Class15_STFT_001/
```

---

## Notes on Old Documentation

**⚠️ The existing `README.md` and `FUNCTIONALITY_CHECKLIST.txt` in the repository are OUTDATED and should be ignored.**

This README is generated exclusively from code analysis and represents the actual implemented behavior as of the provided codebase.

---

## Model Architecture (Inferred)

```
Input: (batch, 12, 1, 101, 193) STFT tensors
       (12, 12) Adjacency matrix (based on channel physical coordinates)

1. Per-channel CNN feature extraction (12 parallel branches)
2. GNN layer (graph convolution over adjacency matrix)
   → Output: (batch, 12, GNN_HIDDEN_DIM) node features
3. Flatten → (batch, 12 * GNN_HIDDEN_DIM)
4. Transformer Encoder (TRANSFORMER_LAYERS layers, TRANSFORMER_NHEAD heads)
   → Temporal modeling
5. Fully connected classifier → (batch, NUM_CLASSES) logits
```

**Adjacency Matrix Construction:**
- Based on Euclidean distance between channel coordinates (`CHANNEL_COORDS`)
- Threshold applied to create binary adjacency
- Defines spatial relationships for GNN message passing

---

## License & Citation

(No license info found in code - add as needed)

For questions or issues, contact the repository maintainer.

---

**END OF README**
