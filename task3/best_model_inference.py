"""
=============================================================================
INFERENCE — Trening z najlepszymi parametrami + generowanie submission
Parametry z Optuny: hidden=256, blocks=4, batch=2048, lr=0.000539
=============================================================================
"""

import os
import gc
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ─────────────────────────────────────────────────────────────────────────────
# 0. KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
TRAIN_PATH  = str(BASE_DIR / "HOURLY_TRAIN_FEATURES.csv")
VAL_PATH    = str(BASE_DIR / "HOURLY_VAL_FEATURES.csv")
TEST_PATH   = str(BASE_DIR / "HOURLY_TEST_FEATURES.csv")
OUTPUT_PATH = str(BASE_DIR / "submission_best.csv")
MODEL_PATH  = str(BASE_DIR / "best_model.pt")

TARGET = "x2_mean"

# ── Najlepsze parametry z Optuny ─────────────────────────────────────────────
BEST_PARAMS = {
    "hidden_dim":    256,
    "n_blocks":      4,
    "dropout":       0.2040533728088605,
    "input_dropout": 0.024407646968955768,
    "lr":            0.0005388108577817234,
    "weight_decay":  1.3726318898045876e-06,
    "batch_size":    2048,
}

# ── Ile epok trenować ─────────────────────────────────────────────────────────
# Poprzednio model osiągnął best na epoce ~19/20
# Dajemy więcej epok żeby w pełni skorzystać z całego trainu
FINAL_EPOCHS = 50
FINAL_PAT    = 10

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── GPU ───────────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print(f"[START] Używam: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
print(f"  Params: {BEST_PARAMS}", flush=True)
print(f"  Epoki: {FINAL_EPOCHS} max | patience={FINAL_PAT}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. WCZYTANIE DANYCH
# ─────────────────────────────────────────────────────────────────────────────

DROP_COLS = [TARGET, "deviceId", "HOUR", "hour", "dso_region", "x2_mean"]


def load_csv_optimized(path: str) -> pd.DataFrame:
    t0 = time.time()
    print(f"\n  [LOAD] {Path(path).name} ...", flush=True)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    hour_col = "HOUR" if "HOUR" in df.columns else "hour"
    if hour_col in df.columns:
        df[hour_col] = pd.to_datetime(df[hour_col])
        df = df.rename(columns={hour_col: "hour"})
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  [LOAD] OK — {len(df):,} wierszy | {mem_mb:.0f} MB | {time.time()-t0:.1f}s",
          flush=True)
    return df


print("\n[1/9] Wczytywanie danych...", flush=True)
df_train = load_csv_optimized(TRAIN_PATH)
df_val   = load_csv_optimized(VAL_PATH)
df_test  = load_csv_optimized(TEST_PATH)
print("[1/9] Wczytywanie gotowe.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CECHY i TARGET
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/9] Przygotowanie cech...", flush=True)

ID_COLS = ["deviceId", "hour"]

feature_cols = [
    c for c in df_train.columns
    if c not in DROP_COLS
    and c not in ID_COLS
    and pd.api.types.is_numeric_dtype(df_train[c])
    and df_train[c].std() > 1e-7
]
print(f"  Liczba cech: {len(feature_cols)}", flush=True)


def prepare_X(df, cols):
    return df[cols].fillna(0).values.astype(np.float32)


def prepare_y(df, target):
    if target not in df.columns:
        return None
    y = df[target].values.astype(np.float32)
    return None if np.isnan(y).all() else y


X_train = prepare_X(df_train, feature_cols)
y_train = prepare_y(df_train, TARGET)
X_val   = prepare_X(df_val,   feature_cols)
X_test  = prepare_X(df_test,  feature_cols)

print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}", flush=True)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)
gc.collect()
print("[2/9] Cechy gotowe.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HOLDOUT Z TRAINU (marzec + kwiecień) — do early stopping
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/9] Tworzenie holdoutu...", flush=True)
df_train["_month"] = df_train["hour"].dt.month
df_train["_year"]  = df_train["hour"].dt.year
holdout_mask = (df_train["_year"] == 2025) & (df_train["_month"].isin([3, 4]))

X_opt_vl = X_train[holdout_mask.values]
y_opt_vl = y_train[holdout_mask.values]

print(f"  holdout: {len(X_opt_vl):,} wierszy (Mar + Apr 2025)", flush=True)
print("[3/9] Holdout gotowy.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ARCHITEKTURA
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ResidualMLP(nn.Module):
    def __init__(self, n_features, hidden_dim, n_blocks, dropout, input_dropout):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for b in self.blocks:
            x = b(x)
        return self.output(x).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNKCJE TRENINGOWE
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, amp_scaler):
    model.train()
    total     = 0.0
    n_batches = len(loader)
    for i, (Xb, yb) in enumerate(loader):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        if USE_AMP:
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(Xb), yb)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * len(yb)
        if (i + 1) % 100 == 0:
            print(f"      batch {i+1}/{n_batches} | loss={total/((i+1)*loader.batch_size):.5f}",
                  flush=True)
    return total / len(loader.dataset)


@torch.no_grad()
def predict_batched(model, X_np, batch_size=65536):
    model.eval()
    preds = []
    for i in range(0, len(X_np), batch_size):
        Xb = torch.tensor(X_np[i:i + batch_size]).to(DEVICE)
        if USE_AMP:
            with torch.amp.autocast("cuda"):
                p = model(Xb).cpu().numpy()
        else:
            p = model(Xb).cpu().numpy()
        preds.append(p)
    return np.concatenate(preds)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRENING
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[4/9] Trening ({FINAL_EPOCHS} epok maks, patience={FINAL_PAT})...", flush=True)

model = ResidualMLP(
    n_features    = X_train.shape[1],
    hidden_dim    = BEST_PARAMS["hidden_dim"],
    n_blocks      = BEST_PARAMS["n_blocks"],
    dropout       = BEST_PARAMS["dropout"],
    input_dropout = BEST_PARAMS["input_dropout"],
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parametry modelu: {n_params:,}", flush=True)

optimizer  = torch.optim.AdamW(model.parameters(),
                               lr=BEST_PARAMS["lr"],
                               weight_decay=BEST_PARAMS["weight_decay"])
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                 optimizer, T_max=FINAL_EPOCHS, eta_min=BEST_PARAMS["lr"] * 0.01)
loss_fn    = nn.L1Loss()
amp_scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size         = BEST_PARAMS["batch_size"],
    shuffle            = True,
    num_workers        = 4,
    pin_memory         = USE_AMP,
    persistent_workers = True,
)

best_mae, best_state, wait = float("inf"), None, 0
t_train_start = time.time()

for epoch in range(1, FINAL_EPOCHS + 1):
    t_ep    = time.time()
    tr_loss = train_epoch(model, loader, optimizer, loss_fn, amp_scaler)
    v_pred  = predict_batched(model, X_opt_vl)
    v_mae   = mean_absolute_error(y_opt_vl, v_pred)
    scheduler.step()

    improved = v_mae < best_mae
    if improved:
        best_mae   = v_mae
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait       = 0
    else:
        wait += 1

    marker = " ✓" if improved else f" (wait {wait}/{FINAL_PAT})"
    print(f"  Epoch {epoch:3d}/{FINAL_EPOCHS} | loss={tr_loss:.5f} | "
          f"val_mae={v_mae:.6f} | best={best_mae:.6f}{marker} | {time.time()-t_ep:.1f}s",
          flush=True)

    if wait >= FINAL_PAT:
        print(f"  Early stop @ epoch {epoch}", flush=True)
        break

model.load_state_dict(best_state)
print(f"\n[4/9] Trening gotowy — best holdout MAE: {best_mae:.6f} | "
      f"łącznie {time.time()-t_train_start:.1f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ZAPIS MODELU
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[5/9] Zapis modelu...", flush=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "best_params":      BEST_PARAMS,
    "feature_cols":     feature_cols,
    "scaler_mean":      scaler.mean_,
    "scaler_scale":     scaler.scale_,
    "holdout_mae":      best_mae,
}, MODEL_PATH)
print(f"[5/9] Model zapisany: {MODEL_PATH}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PREDYKCJA
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[6/9] Predykcja na val ({len(X_val):,} wierszy)...", flush=True)
pred_val_h     = predict_batched(model, X_val).clip(0, 1)
df_val["pred"] = pred_val_h
df_val["year_col"]  = df_val["hour"].dt.year
df_val["month_col"] = df_val["hour"].dt.month
print("[6/9] Val gotowe.", flush=True)

print(f"\n[7/9] Predykcja na test ({len(X_test):,} wierszy)...", flush=True)
pred_test_h     = predict_batched(model, X_test).clip(0, 1)
df_test["pred"] = pred_test_h
df_test["year_col"]  = df_test["hour"].dt.year
df_test["month_col"] = df_test["hour"].dt.month
print("[7/9] Test gotowe.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9. AGREGACJA → SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[8/9] Agregacja i submission...", flush=True)

sub_val = (df_val
           .groupby(["deviceId", "year_col", "month_col"])["pred"].mean()
           .reset_index()
           .rename(columns={"year_col": "year", "month_col": "month", "pred": "prediction"}))

sub_test = (df_test
            .groupby(["deviceId", "year_col", "month_col"])["pred"].mean()
            .reset_index()
            .rename(columns={"year_col": "year", "month_col": "month", "pred": "prediction"}))

submission = pd.concat([sub_val, sub_test], ignore_index=True)

# Uzupełnij brakujące kombinacje
all_devices = df_train["deviceId"].unique()
expected    = pd.DataFrame(
    [(d, 2025, m) for d in all_devices for m in range(5, 11)],
    columns=["deviceId", "year", "month"]
)
submission = expected.merge(submission, on=["deviceId", "year", "month"], how="left")
missing    = submission["prediction"].isna().sum()
print(f"  Brakujących kombinacji: {missing}", flush=True)

dev_med    = df_train.groupby("deviceId")[TARGET].median().rename("med")
submission = submission.merge(dev_med, on="deviceId", how="left")
submission["prediction"] = submission["prediction"].fillna(submission["med"])
submission = submission.drop(columns="med")
submission["prediction"] = submission["prediction"].clip(0, 1)
submission = submission.sort_values(["deviceId", "year", "month"]).reset_index(drop=True)

os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)
submission.to_csv(OUTPUT_PATH, index=False)
print(f"[8/9] Zapisano: {OUTPUT_PATH}", flush=True)

print(f"\n[9/9] PODSUMOWANIE:")
print(f"  Wierszy         : {len(submission):,}  (oczekiwane: 3600)")
print(f"  NaN predykcji   : {submission['prediction'].isna().sum()}")
print(f"  Mean prediction : {submission['prediction'].mean():.4f}")
print(f"  Std  prediction : {submission['prediction'].std():.4f}")
print(f"  Holdout MAE     : {best_mae:.6f}")
print(submission.head(12))
print("\n[DONE]", flush=True)
