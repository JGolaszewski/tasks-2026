# 🔋 Predictive Grid Management for Heat Pump Networks
**Euros Energy Hackathon — Task 3**

> Forecasting monthly electrical grid load for 600 heat pump devices across Poland (May–October 2025), trained on 7 months of winter telemetry (October 2024 – April 2025).

---

## 📊 Results

| Model | Strategy | Holdout MAE | Leaderboard MAE |
|---|---|---|---|
| Monthly MLP | Monthly aggregates → predict | ~0.009 | 0.0225 |
| Monthly LightGBM | Monthly aggregates → predict | ~0.015 | 0.0211 |
| **Hourly Residual MLP** | **Hourly predict → aggregate** | **~0.0075** | **TBD** |

> **Holdout:** March + April 2025 (spring, closer to summer than the rest of training data)

---

## 💡 Core idea — why hourly beats monthly

The competition is an **extrapolation task**: train on winter (Oct 2024 – Apr 2025), predict summer (May–Oct 2025). This creates a severe distribution shift:

| Feature | Training mean | Validation mean | Shift |
|---|---|---|---|
| `imgw_t_mean_C` | 2.7°C | 15.6°C | **6× higher** |
| `heating_degree_days` | 15.3 | 2.4 | **6× lower** |

Monthly models learn seasonal calendar patterns and fail to generalise to temperatures they've never seen.

**Our solution:** train on *hourly* physical measurements instead.

```
Monthly approach (fails):
  calendar features + monthly agg → predict monthly x2

Hourly approach (works):
  t1–t13 per hour + device features → predict x2 per hour → aggregate to monthly mean
```

Internal heat exchanger temperatures (`t4`, `t5`, `t8`, `t12`, `t13`) are **stable between seasons** — they reflect the device's operating state regardless of outdoor temperature. By training on these, the model learns the physics of the device rather than the calendar.

---

## 🗂️ Repository structure

```
├── aggregate_to_hourly.py            # Step 1 — 5-min → 1h aggregation (GPU, cuDF)
├── Feature_engineering_ipynb.ipynb   # Step 2 — Snowflake feature engineering notebook
├── trening_modelnn.py                # Step 3a — Full training with Optuna hyperparameter search
├── best_model_inference.py           # Step 3b — Training with best params (recommended ✅)
├── best_model.pt                     # Saved model weights + scaler + feature names
├── best_params.txt                   # Best hyperparameters found by Optuna
├── submission_hourly_nn.csv          # Final submission file (3600 rows)
└── README.md
```

---

## 🔄 Full pipeline

```
data.csv + devices.csv
(raw 5-min telemetry, ~250M rows)
          │
          ▼  [aggregate_to_hourly.py — GPU / cuDF]
          │  Groups by (deviceId, hour), computes mean/max
          │
dane_task3_aggregated.csv.gz
(~3M hourly rows: x1, x2, t1–t13 per device per hour)
          │
          ▼  [Feature_engineering_ipynb.ipynb — Snowflake Notebook]
          │  Uploads to DANE_TASK3, adds 58 features, splits into train/val/test
          │
HOURLY_TRAIN_FEATURES.csv  (~2.9M rows, Oct 2024 – Apr 2025)
HOURLY_VAL_FEATURES.csv    (~0.84M rows, May – Jun 2025)
HOURLY_TEST_FEATURES.csv   (~1.66M rows, Jul – Oct 2025)
          │
          ▼  [best_model_inference.py — A100/H100 GPU]
          │  Trains Residual MLP, predicts hourly x2, aggregates to monthly mean
          │
submission_best.csv
(3600 rows: 600 devices × 6 months, May–Oct 2025)
```

---

## 🚀 How to reproduce

### Step 1 — Aggregate 5-min data to hourly

Requires `data.csv` and `devices.csv` from the competition dataset and a GPU with RAPIDS cuDF.

```bash
# Google Colab GPU runtime works well for this step
python3 aggregate_to_hourly.py
```

Output: `dane_task3_aggregated.csv.gz`

The script aggregates ~250M 5-minute readings into ~3M hourly records per `(deviceId, hour)`:
- `x1_mean`, `x1_max` — compressor operating frequency
- `x2_mean` — grid load indicator (the prediction target)
- `t1_mean` … `t13_mean` — all 13 temperature sensors (min-max normalised)

Then joins device metadata (`latitude`, `longitude`) from `devices.csv`.

---

### Step 2 — Feature engineering in Snowflake

Open `Feature_engineering_ipynb.ipynb` in a **Snowflake Notebook**.

The first cell uploads `dane_task3_aggregated.csv.gz` to the Snowflake table `DANE_TASK3`. The notebook then adds 58 engineered features and splits data by time period:

| Table | Period | Rows | Notes |
|---|---|---|---|
| `HOURLY_TRAIN_FEATURES` | Oct 2024 – Apr 2025 | ~2.9M | Full feature set incl. `x2_mean` |
| `HOURLY_VAL_FEATURES` | May – Jun 2025 | ~0.84M | `x2_mean` withheld |
| `HOURLY_TEST_FEATURES` | Jul – Oct 2025 | ~1.66M | `x2_mean` withheld |

Export each table as CSV via: *Snowsight → Data → table → ⋮ → Download as CSV*.

**Features added (58 total):**

| Category | Features | Why useful |
|---|---|---|
| Raw sensors | `x1_mean`, `x1_max`, `t1–t13_mean` | Direct device measurements |
| Cyclic time | `hour_sin/cos`, `month_sin/cos`, `dow_sin/cos`, `doy_sin/cos` | Smooth time encoding |
| Time flags | `is_peak_hour`, `is_night`, `is_daylight` | Demand patterns |
| Physical deltas | `load_source_delta`, `load_hex_delta`, `indoor_outdoor`, `air_hex_delta` | Heat exchanger efficiency |
| Climate norms | `imgw_t_mean_C`, `imgw_t_min_C` | IMGW 1991–2020 per DSO region |
| Astronomy | `daylight_hours`, `daylight_norm` | Stable seasonal proxy |
| Geography | `lat_norm`, `lon_norm`, `nearest_centroid_km` | Device location |
| DSO region | `dso_region`, `dso_x2_enc` | Regional grid characteristics |
| Device history | `device_x2_mean_train`, `device_x2_warm/cold`, `device_warm_cold_ratio` | Per-device baseline |
| Temp response | `dev_t1_slope`, `dev_t1_intercept`, `dev_linear_pred` | Linear extrapolation per device |

---

### Step 3 — Train model and generate submission

Place `HOURLY_TRAIN_FEATURES.csv`, `HOURLY_VAL_FEATURES.csv`, `HOURLY_TEST_FEATURES.csv` in the same directory as `best_model_inference.py`.

```bash
# Athena supercomputer (SLURM) — recommended
srun --partition=tutorial --account=tutorial \
     --gpus=1 --time=02:00:00 --mem=64G \
     python3 -u best_model_inference.py

# Locally (CPU — works but slower)
python3 best_model_inference.py
```

The script automatically saves `best_model.pt` and generates `submission_best.csv`.

**To load the pre-trained model without re-training:**

```python
import torch
from best_model_inference import ResidualMLP
from sklearn.preprocessing import StandardScaler

ckpt = torch.load("best_model.pt", map_location="cpu")

model = ResidualMLP(
    n_features    = len(ckpt["feature_cols"]),
    hidden_dim    = ckpt["best_params"]["hidden_dim"],
    n_blocks      = ckpt["best_params"]["n_blocks"],
    dropout       = ckpt["best_params"]["dropout"],
    input_dropout = ckpt["best_params"]["input_dropout"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Restore fitted scaler
scaler = StandardScaler()
scaler.mean_  = ckpt["scaler_mean"]
scaler.scale_ = ckpt["scaler_scale"]

print(f"Loaded model | holdout MAE: {ckpt['holdout_mae']:.6f}")
print(f"Features: {ckpt['feature_cols']}")
```

---

## 🧠 Model architecture

**Residual MLP** — skip connections make deep networks more stable for tabular data than plain stacked layers:

```
Input (58 features)
    │
    ├─ InputDropout (p=0.024)
    │
    ▼  Linear(58 → 256) → LayerNorm → GELU
    │
    ▼  ResidualBlock × 4:
    │    ┌── Linear(256→256) → LayerNorm → GELU ──┐
    │    │   Dropout(p=0.204)                      │
    │    │   Linear(256→256) → LayerNorm           │
    │    └──────── x + f(x) → GELU ───────────────┘
    │
    ▼  Linear(256 → 128) → GELU → Linear(128 → 1)
    │
  Output  (x2 per hour, clipped to [0, 1])
    │
    ▼  GroupBy (deviceId, year, month) → mean
    │
  Submission  (monthly average x2 per device)
```

---

## ⚙️ Hyperparameters

Found via **Optuna TPE sampler** (5 trials on holdout Mar+Apr 2025):

| Parameter | Value |
|---|---|
| `hidden_dim` | 256 |
| `n_blocks` | 4 |
| `dropout` | 0.2041 |
| `input_dropout` | 0.0244 |
| `learning_rate` | 5.39 × 10⁻⁴ |
| `weight_decay` | 1.37 × 10⁻⁶ |
| `batch_size` | 2048 |
| `max_epochs` | 50 |
| `patience` | 10 |

**Training details:**
- Loss function: **L1 (MAE)** — directly optimises the competition metric
- Optimiser: AdamW + CosineAnnealingLR (decays to 1% of initial LR)
- Mixed precision: **AMP float16** — ~2× speedup on A100/H100
- Gradient clipping: norm = 1.0
- Validation: holdout on **Mar + Apr 2025** (spring months, closest to summer)
- Hardware: NVIDIA A100-SXM4-40GB, 40GB VRAM

---

## 📈 Training log (final model)

```
[5/9] Trening (50 epok maks, patience=10)...
  Parametry modelu: 579,073

  Epoch   1/50 | loss=0.02853 | val_mae=0.017114 | best=0.017114 ✓ | 40.4s
  Epoch   2/50 | loss=0.01947 | val_mae=0.014940 | best=0.014940 ✓ | 38.8s
  Epoch   3/50 | loss=0.01735 | val_mae=0.012082 | best=0.012082 ✓ | 39.2s
  Epoch   4/50 | loss=0.01632 | val_mae=0.010941 | best=0.010941 ✓ | 39.0s
  Epoch   5/50 | loss=0.01541 | val_mae=0.010203 | best=0.010203 ✓ | 39.1s
  ...
  Epoch  20/50 | loss=0.01157 | val_mae=0.007512 | best=0.007512 ✓ | 39.4s
  ...
  Early stop @ epoch ~30

[5/9] Finalny model gotowy. Holdout MAE: ~0.0075
```

---

## 📦 Requirements

```bash
# Step 1 — 5-min aggregation (needs GPU + RAPIDS)
pip install cudf-cu12 pandas numpy

# Step 3 — model training
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas numpy optuna
```

**Python:** 3.11.5
