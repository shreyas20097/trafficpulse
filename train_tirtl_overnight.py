from pathlib import Path
import time
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA = Path("artifacts/tirtl_features.csv")
OUT = Path("artifacts/tirtl_training")
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

TRAIN_END = "2026-01-24"
VAL_START = "2026-01-25"
VAL_END = "2026-01-27"
TEST_START = "2026-01-28"
TEST_END = "2026-01-31"


# ============================================================
# HELPERS
# ============================================================

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(name, m):
    print(
        f"{name:<22}"
        f" MAE={m['MAE']:>10.4f}"
        f" RMSE={m['RMSE']:>10.4f}"
        f" R2={m['R2']:>10.4f}"
    )


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("TIRTL OVERNIGHT TRAINING")
print("=" * 80)

if not DATA.exists():
    raise FileNotFoundError(
        f"Could not find {DATA}"
    )

start_total = time.time()

print("\nLoading feature dataset...")

df = pd.read_csv(
    DATA,
    parse_dates=["datetime"]
)

print(f"Rows loaded: {len(df):,}")
print(f"Columns:     {len(df.columns)}")
print(
    f"Date range:  "
    f"{df['datetime'].min()} → "
    f"{df['datetime'].max()}"
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["datetime", "site", "heading"]
).reset_index(drop=True)


# ============================================================
# DATE SPLIT
# ============================================================

train_mask = (
    df["datetime"].dt.date
    <= pd.Timestamp(TRAIN_END).date()
)

val_mask = (
    (df["datetime"].dt.date
     >= pd.Timestamp(VAL_START).date())
    &
    (df["datetime"].dt.date
     <= pd.Timestamp(VAL_END).date())
)

test_mask = (
    (df["datetime"].dt.date
     >= pd.Timestamp(TEST_START).date())
    &
    (df["datetime"].dt.date
     <= pd.Timestamp(TEST_END).date())
)

train = df.loc[train_mask].copy()
val = df.loc[val_mask].copy()
test = df.loc[test_mask].copy()

print("\n" + "=" * 80)
print("CHRONOLOGICAL SPLIT")
print("=" * 80)

print(
    f"TRAIN: {len(train):,} rows | "
    f"{train['datetime'].min()} → {train['datetime'].max()}"
)

print(
    f"VAL:   {len(val):,} rows | "
    f"{val['datetime'].min()} → {val['datetime'].max()}"
)

print(
    f"TEST:  {len(test):,} rows | "
    f"{test['datetime'].min()} → {test['datetime'].max()}"
)


# ============================================================
# FEATURES
# ============================================================

# These are intentionally explicit.
#
# We DO NOT feed:
#   date
#   time_bin
#   datetime
#
# directly into the tree models.
#
# site and heading are categorical identifiers and will be
# one-hot encoded below.

numeric_features = [
    "bicycle",
    "error",
    "heavy_articulated",
    "motorcycle",
    "rigid_truck_bus",
    "short_vehicle",
    "short_vehicle_towing",
    "unknown",

    "mean_speed",
    "total_motorized",
    "total_all",

    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "time_sin",
    "time_cos",

    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",

    "rolling_mean_4",
    "rolling_std_4",
    "rolling_mean_8",
    "rolling_std_8",

    "motorcycle_share",
    "short_vehicle_share",
    "short_vehicle_towing_share",
    "rigid_truck_bus_share",
    "heavy_articulated_share",
]

categorical_features = [
    "site",
    "heading",
]

TARGET = "target_next_15min"

FEATURES = numeric_features + categorical_features


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    c for c in FEATURES + [TARGET]
    if c not in df.columns
]

if missing_features:
    raise RuntimeError(
        "Missing expected columns:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# ONE-HOT ENCODING
# ============================================================

print("\nPreparing features...")

X_all = df[FEATURES].copy()

# Treat identifiers as categorical.
X_all["site"] = X_all["site"].astype(str)
X_all["heading"] = X_all["heading"].astype(str)

X_all = pd.get_dummies(
    X_all,
    columns=["site", "heading"],
    dtype=np.float32
)

# Make all numeric
X_all = X_all.astype(np.float32)

y_all = df[TARGET].astype(np.float32)

X_train = X_all.loc[train.index]
X_val = X_all.loc[val.index]
X_test = X_all.loc[test.index]

y_train = y_all.loc[train.index]
y_val = y_all.loc[val.index]
y_test = y_all.loc[test.index]

print(f"Model features: {X_train.shape[1]}")
print(f"Training rows:  {len(X_train):,}")
print(f"Validation:     {len(X_val):,}")
print(f"Test:           {len(X_test):,}")


# ============================================================
# REMOVE NON-FINITE VALUES
# ============================================================

# Replace infinities.
X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)
X_val = X_val.replace(
    [np.inf, -np.inf],
    np.nan
)
X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# Tree models can tolerate NaNs, but keeping the same finite
# representation makes the experiment easier to reproduce.
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 80)
print("BASELINE")
print("=" * 80)

# lag_1 is the previous 15-minute observed demand.
# For the current observation, it is the demand from the
# immediately preceding interval.
#
# The persistence prediction therefore uses the current
# demand (total_motorized), which is available at prediction
# time.

baseline_pred_val = val["total_motorized"].to_numpy()
baseline_pred_test = test["total_motorized"].to_numpy()

baseline_val = metrics(
    y_val,
    baseline_pred_val
)

baseline_test = metrics(
    y_test,
    baseline_pred_test
)

print_metrics(
    "Persistence VAL",
    baseline_val
)

print_metrics(
    "Persistence TEST",
    baseline_test
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST")
print("=" * 80)

rf_start = time.time()

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=18,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

print("Training Random Forest...")

rf.fit(
    X_train,
    y_train
)

rf_time = time.time() - rf_start

rf_val_pred = rf.predict(X_val)
rf_test_pred = rf.predict(X_test)

rf_val = metrics(
    y_val,
    rf_val_pred
)

rf_test = metrics(
    y_test,
    rf_test_pred
)

print_metrics(
    "Random Forest VAL",
    rf_val
)

print_metrics(
    "Random Forest TEST",
    rf_test
)

print(f"Training time: {rf_time / 60:.2f} minutes")

# Save model
import joblib

joblib.dump(
    rf,
    OUT / "random_forest.joblib"
)


# ============================================================
# XGBOOST
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST")
print("=" * 80)

xgb_start = time.time()

xgb = XGBRegressor(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.04,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,

    objective="reg:squarederror",
    eval_metric="rmse",

    tree_method="hist",

    device="cuda",
    random_state=RANDOM_STATE,
)

print("Training XGBoost...")

xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

xgb_time = time.time() - xgb_start

xgb_val_pred = xgb.predict(X_val)
xgb_test_pred = xgb.predict(X_test)

xgb_val = metrics(
    y_val,
    xgb_val_pred
)

xgb_test = metrics(
    y_test,
    xgb_test_pred
)

print_metrics(
    "XGBoost VAL",
    xgb_val
)

print_metrics(
    "XGBoost TEST",
    xgb_test
)

print(f"Training time: {xgb_time / 60:.2f} minutes")

joblib.dump(
    xgb,
    OUT / "xgboost.joblib"
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

results = pd.DataFrame([
    {
        "model": "Persistence",
        "split": "validation",
        **baseline_val,
    },
    {
        "model": "Random Forest",
        "split": "validation",
        **rf_val,
    },
    {
        "model": "XGBoost",
        "split": "validation",
        **xgb_val,
    },
    {
        "model": "Persistence",
        "split": "test",
        **baseline_test,
    },
    {
        "model": "Random Forest",
        "split": "test",
        **rf_test,
    },
    {
        "model": "XGBoost",
        "split": "test",
        **xgb_test,
    },
])

print(results.to_string(index=False))

results.to_csv(
    OUT / "model_comparison.csv",
    index=False
)


# ============================================================
# IMPROVEMENT OVER BASELINE
# ============================================================

def improvement(baseline, model):
    return {
        "MAE_improvement_pct": (
            (baseline["MAE"] - model["MAE"])
            / baseline["MAE"]
            * 100
        ),
        "RMSE_improvement_pct": (
            (baseline["RMSE"] - model["RMSE"])
            / baseline["RMSE"]
            * 100
        ),
    }


improvements = {
    "Random Forest": {
        "validation": improvement(
            baseline_val,
            rf_val
        ),
        "test": improvement(
            baseline_test,
            rf_test
        ),
    },
    "XGBoost": {
        "validation": improvement(
            baseline_val,
            xgb_val
        ),
        "test": improvement(
            baseline_test,
            xgb_test
        ),
    },
}

print("\n" + "=" * 80)
print("IMPROVEMENT OVER PERSISTENCE")
print("=" * 80)

for model, splits in improvements.items():

    for split, values in splits.items():

        print(
            f"{model:<15} {split:<12}"
            f" MAE: {values['MAE_improvement_pct']:>8.2f}%"
            f" RMSE: {values['RMSE_improvement_pct']:>8.2f}%"
        )


save_json(
    improvements,
    OUT / "baseline_improvement.json"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

feature_names = X_train.columns

rf_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_,
}).sort_values(
    "importance",
    ascending=False
)

xgb_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": xgb.feature_importances_,
}).sort_values(
    "importance",
    ascending=False
)

rf_importance.to_csv(
    OUT / "random_forest_feature_importance.csv",
    index=False
)

xgb_importance.to_csv(
    OUT / "xgboost_feature_importance.csv",
    index=False
)

print("\nTop Random Forest features:")
print(
    rf_importance.head(20).to_string(
        index=False
    )
)

print("\nTop XGBoost features:")
print(
    xgb_importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = test[
    [
        "datetime",
        "date",
        "time_bin",
        "site",
        "heading",
        "total_motorized",
        TARGET,
    ]
].copy()

predictions["persistence_prediction"] = (
    baseline_pred_test
)

predictions["random_forest_prediction"] = (
    rf_test_pred
)

predictions["xgboost_prediction"] = (
    xgb_test_pred
)

predictions["rf_error"] = (
    predictions[TARGET]
    - predictions["random_forest_prediction"]
)

predictions["xgb_error"] = (
    predictions[TARGET]
    - predictions["xgboost_prediction"]
)

predictions.to_csv(
    OUT / "test_predictions.csv",
    index=False
)


# ============================================================
# PER-SITE / HEADING TEST PERFORMANCE
# ============================================================

print("\nCalculating per-site/heading performance...")

eval_df = predictions.copy()

rows = []

for (site, heading), g in eval_df.groupby(
    ["site", "heading"]
):

    y = g[TARGET].to_numpy()

    row = {
        "site": site,
        "heading": heading,
        "n": len(g),
    }

    for name, pred_col in [
        ("persistence", "persistence_prediction"),
        ("random_forest", "random_forest_prediction"),
        ("xgboost", "xgboost_prediction"),
    ]:

        p = g[pred_col].to_numpy()

        row[f"{name}_MAE"] = (
            mean_absolute_error(y, p)
        )

        row[f"{name}_RMSE"] = np.sqrt(
            mean_squared_error(y, p)
        )

    rows.append(row)

site_results = pd.DataFrame(rows)

site_results.to_csv(
    OUT / "site_heading_performance.csv",
    index=False
)


# ============================================================
# DAILY TEST PERFORMANCE
# ============================================================

rows = []

for date, g in eval_df.groupby("date"):

    y = g[TARGET].to_numpy()

    row = {
        "date": date,
        "n": len(g),
    }

    for name, pred_col in [
        ("persistence", "persistence_prediction"),
        ("random_forest", "random_forest_prediction"),
        ("xgboost", "xgboost_prediction"),
    ]:

        p = g[pred_col].to_numpy()

        row[f"{name}_MAE"] = (
            mean_absolute_error(y, p)
        )

        row[f"{name}_RMSE"] = np.sqrt(
            mean_squared_error(y, p)
        )

    rows.append(row)

daily_results = pd.DataFrame(rows)

daily_results.to_csv(
    OUT / "daily_performance.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

total_time = time.time() - start_total

summary = {
    "dataset": str(DATA),
    "train_end": TRAIN_END,
    "validation_start": VAL_START,
    "validation_end": VAL_END,
    "test_start": TEST_START,
    "test_end": TEST_END,

    "rows": {
        "total": len(df),
        "train": len(train),
        "validation": len(val),
        "test": len(test),
    },

    "features": X_train.shape[1],

    "models": {
        "persistence_validation": baseline_val,
        "persistence_test": baseline_test,

        "random_forest_validation": rf_val,
        "random_forest_test": rf_test,

        "xgboost_validation": xgb_val,
        "xgboost_test": xgb_test,
    },

    "training_time_seconds": {
        "random_forest": rf_time,
        "xgboost": xgb_time,
        "total": total_time,
    },

    "baseline_improvement": improvements,
}

save_json(
    summary,
    OUT / "training_summary.json"
)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(
    f"Total runtime: "
    f"{total_time / 60:.2f} minutes"
)

print(
    f"\nResults saved in:\n"
    f"  {OUT}"
)

print(
    "\nMain files:"
    "\n  model_comparison.csv"
    "\n  training_summary.json"
    "\n  test_predictions.csv"
    "\n  site_heading_performance.csv"
    "\n  daily_performance.csv"
    "\n  random_forest_feature_importance.csv"
    "\n  xgboost_feature_importance.csv"
    "\n  random_forest.joblib"
    "\n  xgboost.joblib"
)

