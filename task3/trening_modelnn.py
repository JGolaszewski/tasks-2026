"""
=============================================================================
HACKATHON MODEL — Sieć neuronowa na danych godzinowych (H100/A100 GPU)
Wersja: ~10-15 minut (FAST_MODE=True) lub ~2-3h (FAST_MODE=False)
=============================================================================
"""

import os
import gc
import time
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# 0. KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_PATH  = "/net/tscratch/people/tutorial241/task3_workspace/HOURLY_TRAIN_FEATURES.csv"
VAL_PATH    = "/net/tscratch/people/tutorial241/task3_workspace/HOURLY_VAL_FEATURES.csv"
TEST_PATH   = "/net/tscratch/people/tutorial241/task3_workspace/HOURLY_TEST_FEATURES.csv"
OUTPUT_PATH = "/net/tscratch/people/tutorial241/task3_workspace/submission_hourly_nn.csv"

TARGET = "x2_mean"

# ── Tryb szybki vs pełny ─────────────────────────────────────────────────────
FAST_MODE = True   # True = ~10-15 min | False = pełny trening (~2-3h)

if FAST_MODE:
    N_TRIALS      = 5
    OPTUNA_EPOCHS = 8
    OPTUNA_PAT    = 3
    FINAL_EPOCHS  = 20
    FINAL_PAT     = 5
else:
    N_TRIALS      = 75
    OPTUNA_EPOCHS = 60
    OPTUNA_PAT    = 10
    FINAL_EPOCHS  = 150
    FINAL_PAT     = 15

# ── GPU ───────────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print(f"[START] Używam: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
print(f"  FAST_MODE={FAST_MODE} | trials={N_TRIALS} | optuna_epochs={OPTUNA_EPOCHS} | final_epochs={FINAL_EPOCHS}", flush=True)


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


print("\n[1/11] Wczytywanie danych...", flush=True)
df_train = load_csv_optimized(TRAIN_PATH)
df_val   = load_csv_optimized(VAL_PATH)
df_test  = load_csv_optimized(TEST_PATH)
print("[1/11] Wczytywanie gotowe.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CECHY i TARGET
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/11] Przygotowanie cech...", flush=True)

ID_COLS = ["deviceId", "hour"]

feature_cols = [
    c for c in df_train.columns
    if c not in DROP_COLS
    and c not in ID_COLS
    and pd.api.types.is_numeric_dtype(df_train[c])
    and df_train[c].std() > 1e-7
]
print(f"  Liczba cech: {len(feature_cols)}", flush=True)
print(f"  Cechy: {feature_cols}", flush=True)


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
y_val   = prepare_y(df_val,   TARGET)
X_test  = prepare_X(df_test,  feature_cols)

print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}", flush=True)
print(f"  X_val  : {X_val.shape}    y_val: {'withheld (NaN)' if y_val is None else y_val.shape}", flush=True)
print(f"  X_test : {X_test.shape}", flush=True)

print("  Skalowanie StandardScaler...", flush=True)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)
gc.collect()
print("[2/11] Cechy gotowe.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HOLDOUT Z TRAINU (marzec + kwiecień)
# y_val jest withheld — walidujemy na ostatnich 2 miesiącach trainu
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/11] Tworzenie holdoutu...", flush=True)
df_train["_month"] = df_train["hour"].dt.month
df_train["_year"]  = df_train["hour"].dt.year
holdout_mask = (df_train["_year"] == 2025) & (df_train["_month"].isin([3, 4]))

X_opt_tr = X_train[~holdout_mask.values]
y_opt_tr = y_train[~holdout_mask.values]
X_opt_vl = X_train[holdout_mask.values]
y_opt_vl = y_train[holdout_mask.values]

print(f"  fit:     {len(X_opt_tr):,} wierszy (Oct 2024 – Feb 2025)", flush=True)
print(f"  holdout: {len(X_opt_vl):,} wierszy (Mar + Apr 2025)", flush=True)
print("[3/11] Holdout gotowy.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ARCHITEKTURA — Residual MLP
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
    total    = 0.0
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

        # Print co 100 batchy żeby widać że żyje
        if (i + 1) % 100 == 0:
            print(f"      batch {i+1}/{n_batches} | loss={total / ((i+1)*loader.batch_size):.5f}",
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


def fit_model(params, X_tr, y_tr, X_v, y_v,
              max_epochs, patience, verbose=True, label=""):
    t_start = time.time()
    model = ResidualMLP(
        n_features    = X_tr.shape[1],
        hidden_dim    = params["hidden_dim"],
        n_blocks      = params["n_blocks"],
        dropout       = params["dropout"],
        input_dropout = params["input_dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model: hidden={params['hidden_dim']} blocks={params['n_blocks']} "
              f"params={n_params:,} | batch={params['batch_size']} lr={params['lr']:.5f}",
              flush=True)

    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=params["lr"],
                                   weight_decay=params["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=max_epochs, eta_min=params["lr"] * 0.01)
    loss_fn    = nn.L1Loss()
    amp_scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size         = params["batch_size"],
        shuffle            = True,
        num_workers        = 4,
        pin_memory         = USE_AMP,
        persistent_workers = True,
    )

    best_mae, best_state, wait = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        tr_loss = train_epoch(model, loader, optimizer, loss_fn, amp_scaler)
        v_pred  = predict_batched(model, X_v)
        v_mae   = mean_absolute_error(y_v, v_pred)
        scheduler.step()

        improved = v_mae < best_mae
        if improved:
            best_mae   = v_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1

        # Print po KAŻDEJ epoce
        marker = " ✓" if improved else f" (wait {wait}/{patience})"
        print(f"  {label}Epoch {epoch:3d}/{max_epochs} | "
              f"loss={tr_loss:.5f} | val_mae={v_mae:.6f} | "
              f"best={best_mae:.6f}{marker} | {time.time()-t_ep:.1f}s",
              flush=True)

        if wait >= patience:
            print(f"  Early stop @ epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    print(f"  Trening zakończony — best_mae={best_mae:.6f} | "
          f"łącznie {time.time()-t_start:.1f}s", flush=True)
    return model, best_mae


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTUNA
# ─────────────────────────────────────────────────────────────────────────────

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)


def objective(trial):
    params = {
        "hidden_dim":    trial.suggest_categorical("hidden_dim",  [256, 512]),
        "n_blocks":      trial.suggest_int("n_blocks",            2, 4),
        "dropout":       trial.suggest_float("dropout",           0.05, 0.4),
        "input_dropout": trial.suggest_float("input_dropout",     0.0, 0.2),
        "lr":            trial.suggest_float("lr",                1e-4, 3e-3, log=True),
        "weight_decay":  trial.suggest_float("weight_decay",      1e-6, 1e-2, log=True),
        "batch_size":    trial.suggest_categorical("batch_size",  [2048, 4096]),
    }

    completed = [t.value for t in study.trials if t.value is not None]
    best_so_far = min(completed) if completed else float("inf")

    print(f"\n--- Trial {trial.number+1}/{N_TRIALS} "
          f"(best so far: {best_so_far:.6f}) ---", flush=True)

    _, mae = fit_model(
        params, X_opt_tr, y_opt_tr, X_opt_vl, y_opt_vl,
        max_epochs=OPTUNA_EPOCHS, patience=OPTUNA_PAT,
        verbose=True, label=f"[T{trial.number+1}] ",
    )

    print(f"--- Trial {trial.number+1}/{N_TRIALS} KONIEC | mae={mae:.6f} ---",
          flush=True)
    return mae


print(f"\n[4/11] Optuna — {N_TRIALS} trialsów...", flush=True)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f"\n[4/11] Optuna zakończona.", flush=True)
print(f"  Najlepszy val_mae : {study.best_value:.6f}", flush=True)
print(f"  Najlepsze params  : {study.best_params}", flush=True)
best_params = study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 7. FINALNY MODEL — CAŁE DANE TRENINGOWE
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[5/11] Finalny model ({FINAL_EPOCHS} epok maks)...", flush=True)
final_model, final_mae = fit_model(
    best_params,
    X_train, y_train,    # cały train
    X_opt_vl, y_opt_vl,  # walidacja na holdoucie — NIE na y_val (withheld)!
    max_epochs=FINAL_EPOCHS,
    patience=FINAL_PAT,
    verbose=True,
    label="[FINAL] ",
)
print(f"[5/11] Finalny model gotowy. Holdout MAE: {final_mae:.6f}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PREDYKCJA NA VAL
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[6/11] Predykcja na val ({len(X_val):,} wierszy)...", flush=True)
pred_val_h          = predict_batched(final_model, X_val).clip(0, 1)
df_val["pred"]      = pred_val_h
df_val["year_col"]  = df_val["hour"].dt.year
df_val["month_col"] = df_val["hour"].dt.month
print(f"[6/11] Predykcja val gotowa.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9. PREDYKCJA NA TEST
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[7/11] Predykcja na test ({len(X_test):,} wierszy)...", flush=True)
pred_test_h          = predict_batched(final_model, X_test).clip(0, 1)
df_test["pred"]      = pred_test_h
df_test["year_col"]  = df_test["hour"].dt.year
df_test["month_col"] = df_test["hour"].dt.month
print(f"[7/11] Predykcja test gotowa.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 10. AGREGACJA GODZINOWYCH PREDYKCJI → MIESIĘCZNE
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[8/11] Agregacja do miesięcznych predykcji...", flush=True)

sub_val = (
    df_val
    .groupby(["deviceId", "year_col", "month_col"])["pred"]
    .mean().reset_index()
    .rename(columns={"year_col": "year", "month_col": "month", "pred": "prediction"})
)
sub_test = (
    df_test
    .groupby(["deviceId", "year_col", "month_col"])["pred"]
    .mean().reset_index()
    .rename(columns={"year_col": "year", "month_col": "month", "pred": "prediction"})
)

submission = pd.concat([sub_val, sub_test], ignore_index=True)
print(f"  Val  rows : {len(sub_val):,}", flush=True)
print(f"  Test rows : {len(sub_test):,}", flush=True)
print(f"  Łącznie   : {len(submission):,}", flush=True)
print(f"[8/11] Agregacja gotowa.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11. UZUPEŁNIENIE BRAKUJĄCYCH KOMBINACJI → 3600 WIERSZY
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[9/11] Uzupełnianie brakujących kombinacji...", flush=True)

all_devices = df_train["deviceId"].unique()
expected    = pd.DataFrame(
    [(d, 2025, m) for d in all_devices for m in range(5, 11)],
    columns=["deviceId", "year", "month"]
)
submission  = expected.merge(submission, on=["deviceId", "year", "month"], how="left")
missing     = submission["prediction"].isna().sum()
print(f"  Brakujących kombinacji: {missing}", flush=True)

dev_med     = df_train.groupby("deviceId")[TARGET].median().rename("med")
submission  = submission.merge(dev_med, on="deviceId", how="left")
submission["prediction"] = submission["prediction"].fillna(submission["med"])
submission  = submission.drop(columns="med")
submission["prediction"] = submission["prediction"].clip(0, 1)
submission  = submission.sort_values(["deviceId", "year", "month"]).reset_index(drop=True)
print(f"[9/11] Uzupełnianie gotowe.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 12. ZAPIS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[10/11] Zapis submission...", flush=True)
os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)
submission.to_csv(OUTPUT_PATH, index=False)
print(f"[10/11] Zapisano: {OUTPUT_PATH}", flush=True)

print(f"\n[11/11] PODSUMOWANIE:")
print(f"  Wierszy         : {len(submission):,}  (oczekiwane: 3600)")
print(f"  NaN predykcji   : {submission['prediction'].isna().sum()}")
print(f"  Mean prediction : {submission['prediction'].mean():.4f}")
print(f"  Std  prediction : {submission['prediction'].std():.4f}")
print(f"  Holdout MAE     : {final_mae:.6f}")
print(submission.head(12))
print("\n[DONE]", flush=True)
