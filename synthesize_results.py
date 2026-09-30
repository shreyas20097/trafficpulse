#!/usr/bin/env python3
"""
TrafficPulse results synthesis.

Reads the frozen experimental evidence and produces the final results
package for the report/presentation (no new simulations):

Inputs:
  artifacts/tirtl_training/model_comparison.csv
  artifacts/evaluation/trafficpulse_evaluation.csv (+ .json summary)
  artifacts/experiments/volume/volume_scenarios.csv      (40 runs)
  artifacts/experiments/composition/composition_scenarios.csv  (16 runs)
  artifacts/experiments/capacity/capacity_scenarios.csv  (16 runs)

Outputs (artifacts/results/):
  master_experiments.csv      standardized 72-run dataset
  table1_ml_performance.csv   Persistence / RF / XGBoost (test split)
  table2_forecast_validation.csv  8 periods: actual/predicted demand + errors
  table3_volume.csv           volume scenarios (long format)
  table4_infra_composition.csv  composition + capacity in one table
  summary_stats.json          headline numbers for the report
  fig1_model_comparison.png
  fig2_actual_vs_predicted.png
  fig3_demand_vs_travel.png
  fig4_demand_vs_queue.png
  fig5_composition_effect.png
  fig6_capacity_effect.png

Run with the ML venv (needs pandas + matplotlib):
  .venv/bin/python synthesize_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
RESULT_DIR = PROJECT / "artifacts/results"
FIG_DIR = RESULT_DIR / "figures"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TIME_ORDER = ["06:30", "08:30", "10:45", "12:15",
              "14:30", "17:15", "18:45", "20:00"]
VOLUME_ORDER = ["low", "predicted", "baseline", "high", "very_high"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------- inputs

model_comp = pd.read_csv(PROJECT / "artifacts/tirtl_training/model_comparison.csv")
test_metrics = model_comp[model_comp["split"] == "test"].set_index("model")

evaluation = pd.read_csv(PROJECT / "artifacts/evaluation/trafficpulse_evaluation.csv")
with open(PROJECT / "artifacts/evaluation/trafficpulse_evaluation.json") as f:
    eval_summary = json.load(f)["summary"]

volume = pd.read_csv(PROJECT / "artifacts/experiments/volume/volume_scenarios.csv")
composition = pd.read_csv(
    PROJECT / "artifacts/experiments/composition/composition_scenarios.csv")
capacity = pd.read_csv(
    PROJECT / "artifacts/experiments/capacity/capacity_scenarios.csv")

assert len(volume) == 40, len(volume)
assert len(composition) == 16, len(composition)
assert len(capacity) == 16, len(capacity)
for df in (volume, composition, capacity):
    assert (df["completion_rate_pct"] == 100.0).all()
    assert (df["vehicles_reported"] == df["demand_veh_15min"]).all()


# ------------------------------------------------------- master experiment table

def standardize(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["experiment"] = experiment
    out["date"] = df["date"]
    out["time"] = df["time"]
    out["scenario"] = df["scenario"] if "scenario" in df else (
        df["lanes_per_approach"].map({1: "1-lane", 2: "2-lane"}))
    out["demand_veh_15min"] = df["demand_veh_15min"]
    out["lanes_per_approach"] = df.get("lanes_per_approach", 1)
    if "composition_source" in df:
        out["composition"] = df["composition_source"]
    else:
        out["composition"] = "TIRTL audited hourly"
    # Per-class shares when the source experiment records them
    # (composition experiment); NaN otherwise.
    for vtype in ("short", "towing", "rigid", "heavy", "motorcycle"):
        src = f"{vtype}_share"
        out[src] = df[src] if src in df.columns else np.nan
    out["mean_travel_time_s"] = df["mean_travel_time_s"]
    out["mean_waiting_time_s"] = df["mean_waiting_time_s"]
    out["mean_time_loss_s"] = df["mean_time_loss_s"]
    out["mean_queue_vehicles"] = df["mean_queue_vehicles"]
    out["max_queue_vehicles"] = df["max_queue_vehicles"]
    out["vehicles_arrived"] = df["vehicles_arrived"]
    out["completion_rate_pct"] = df["completion_rate_pct"]
    return out


master = pd.concat([
    standardize(volume, "volume"),
    standardize(composition, "composition"),
    standardize(capacity, "capacity"),
], ignore_index=True)
assert len(master) == 72, len(master)
master.to_csv(RESULT_DIR / "master_experiments.csv", index=False)


# ------------------------------------------------------------------ Table 1: ML

table1 = test_metrics.loc[
    ["Persistence", "Random Forest", "XGBoost"], ["MAE", "RMSE", "R2"]]
table1.columns = ["MAE", "RMSE", "R²"]
table1.to_csv(RESULT_DIR / "table1_ml_performance.csv")


# ------------------------------------------------------- Table 2: forecast check

table2 = evaluation[[
    "time",
    "actual_representative_demand_veh_15min",
    "predicted_representative_demand_veh_15min",
    "representative_abs_error",
    "representative_pct_error",
]].copy()
table2.columns = ["time", "actual_demand", "predicted_demand",
                  "absolute_error", "percentage_error"]
table2.to_csv(RESULT_DIR / "table2_forecast_validation.csv", index=False)


# -------------------------------------------------------------- Table 3: volume

table3 = volume[["time", "scenario", "demand_veh_15min",
                 "mean_travel_time_s", "mean_waiting_time_s",
                 "mean_time_loss_s", "mean_queue_vehicles",
                 "max_queue_vehicles", "vehicles_arrived",
                 "completion_rate_pct"]].copy()
table3["time"] = pd.Categorical(table3["time"], TIME_ORDER)
table3["scenario"] = pd.Categorical(table3["scenario"], VOLUME_ORDER)
table3 = table3.sort_values(["time", "scenario"]).reset_index(drop=True)
table3.to_csv(RESULT_DIR / "table3_volume.csv", index=False)


# ----------------------------------------------- Table 4: composition + capacity

comp_rows = standardize(composition, "composition")
cap_rows = standardize(capacity, "capacity")
table4 = pd.concat([comp_rows, cap_rows], ignore_index=True)
table4.to_csv(RESULT_DIR / "table4_infra_composition.csv", index=False)


# ------------------------------------------------------------ summary statistics

cap1 = capacity[capacity["lanes_per_approach"] == 1].set_index("time")
cap2 = capacity[capacity["lanes_per_approach"] == 2].set_index("time")
travel_reduction = (cap1["mean_travel_time_s"] - cap2["mean_travel_time_s"])
queue_reduction = (cap1["mean_queue_vehicles"] - cap2["mean_queue_vehicles"])

comp_n = composition[composition["scenario"] == "normal"].set_index("time")
comp_h = composition[composition["scenario"] == "heavy"].set_index("time")
heavy_d_travel = comp_h["mean_travel_time_s"] - comp_n["mean_travel_time_s"]
heavy_d_wait = comp_h["mean_waiting_time_s"] - comp_n["mean_waiting_time_s"]
heavy_d_queue = comp_h["mean_queue_vehicles"] - comp_n["mean_queue_vehicles"]

summary = {
    "ml_test": {
        model: {"MAE": float(test_metrics.loc[model, "MAE"]),
                "RMSE": float(test_metrics.loc[model, "RMSE"]),
                "R2": float(test_metrics.loc[model, "R2"])}
        for model in ["Persistence", "Random Forest", "XGBoost"]
    },
    "end_to_end": {
        "source_stream_mae": eval_summary["source_stream_mae"],
        "source_stream_rmse": eval_summary["source_stream_rmse"],
        "mean_source_pct_error": eval_summary["mean_source_pct_error"],
        "representative_demand_mae":
            eval_summary["representative_demand_mae"],
        "representative_demand_rmse":
            eval_summary["representative_demand_rmse"],
        "mean_representative_pct_error":
            eval_summary["mean_representative_pct_error"],
    },
    "volume_demand_queue_pearson_r": float(
        volume["demand_veh_15min"].corr(volume["mean_queue_vehicles"])),
    "volume_demand_travel_pearson_r": float(
        volume["demand_veh_15min"].corr(volume["mean_travel_time_s"])),
    "capacity_travel_reduction_s": {
        "mean": float(travel_reduction.mean()),
        "min": float(travel_reduction.min()),
        "max": float(travel_reduction.max()),
    },
    "capacity_mean_queue_reduction_veh": {
        "mean": float(queue_reduction.mean()),
        "min": float(queue_reduction.min()),
        "max": float(queue_reduction.max()),
    },
    "heavy_composition_effect_mean": {
        "travel_time_s": float(heavy_d_travel.mean()),
        "waiting_time_s": float(heavy_d_wait.mean()),
        "mean_queue_veh": float(heavy_d_queue.mean()),
    },
    "n_runs": {"volume": 40, "composition": 16, "capacity": 16, "total": 72},
    "all_runs_completed_100pct": True,
    # Flat aliases consumed by streamlit_app.py (same values, same source).
    "xgboost_test_mae": float(test_metrics.loc["XGBoost", "MAE"]),
    "xgboost_test_r2": float(test_metrics.loc["XGBoost", "R2"]),
    "mean_source_pct_error": eval_summary["mean_source_pct_error"],
    "demand_queue_pearson_r": float(
        volume["demand_veh_15min"].corr(volume["mean_queue_vehicles"])),
    "mean_two_lane_travel_reduction_s": float(travel_reduction.mean()),
    "mean_heavy_travel_delta_s": float(heavy_d_travel.mean()),
}
with open(RESULT_DIR / "summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)


# ------------------------------------------------------------------- figures

# Figure 1 — Model comparison (test MAE/RMSE).
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
models = ["Persistence", "Random Forest", "XGBoost"]
for ax, metric in zip(axes, ["MAE", "RMSE"]):
    ax.bar(models, [test_metrics.loc[m, metric] for m in models])
    ax.set_title(f"Test {metric} (veh / 15 min)")
    ax.set_ylabel(metric)
    for i, m in enumerate(models):
        ax.text(i, test_metrics.loc[m, metric],
                f"{test_metrics.loc[m, metric]:.1f}",
                ha="center", va="bottom", fontsize=9)
fig.suptitle("Figure 1 — ML model comparison (held-out test set)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_model_comparison.png")
plt.close(fig)

# Figure 2 — Actual vs predicted demand across the 8 periods.
x = np.arange(len(TIME_ORDER))
ev = evaluation.set_index("time").loc[TIME_ORDER]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(x, ev["actual_representative_demand_veh_15min"],
        marker="o", label="Actual demand")
ax.plot(x, ev["predicted_representative_demand_veh_15min"],
        marker="s", label="TrafficPulse predicted demand")
ax.set_xticks(x, TIME_ORDER)
ax.set_xlabel("Representative period (17 Jan 2026)")
ax.set_ylabel("Vehicles / 15 min")
ax.set_title("Figure 2 — Actual vs predicted demand (representative intersection)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_actual_vs_predicted.png")
plt.close(fig)

# Figure 3 — Demand vs travel time (volume experiment).
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(volume["demand_veh_15min"], volume["mean_travel_time_s"], alpha=0.7)
coef = np.polyfit(volume["demand_veh_15min"],
                  volume["mean_travel_time_s"], 1)
xs = np.array([volume["demand_veh_15min"].min(),
               volume["demand_veh_15min"].max()])
ax.plot(xs, coef[0] * xs + coef[1], linestyle="--")
ax.set_xlabel("Demand (vehicles / 15 min)")
ax.set_ylabel("Mean travel time (s)")
ax.set_title("Figure 3 — Demand vs mean travel time (volume experiment)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_demand_vs_travel.png")
plt.close(fig)

# Figure 4 — Demand vs mean queue (volume experiment).
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(volume["demand_veh_15min"], volume["mean_queue_vehicles"], alpha=0.7)
coef = np.polyfit(volume["demand_veh_15min"],
                  volume["mean_queue_vehicles"], 1)
ax.plot(xs, coef[0] * xs + coef[1], linestyle="--")
ax.set_xlabel("Demand (vehicles / 15 min)")
ax.set_ylabel("Mean queue (vehicles)")
ax.set_title("Figure 4 — Demand vs mean queue (volume experiment)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_demand_vs_queue.png")
plt.close(fig)

# Figure 5 — Composition effect: normal vs heavy travel time per period.
cn = comp_n.loc[TIME_ORDER, "mean_travel_time_s"].values
ch = comp_h.loc[TIME_ORDER, "mean_travel_time_s"].values
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.35
ax.bar(x - w / 2, cn, w, label="Normal (TIRTL)")
ax.bar(x + w / 2, ch, w, label="Heavy (75/2/15/7/1)")
ax.set_xticks(x, TIME_ORDER)
ax.set_xlabel("Representative period")
ax.set_ylabel("Mean travel time (s)")
ax.set_title("Figure 5 — Vehicle-composition effect (same demand, normal vs heavy)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_composition_effect.png")
plt.close(fig)

# Figure 6 — Capacity effect: travel time + mean queue, 1 vs 2 lanes.
c1t = cap1.loc[TIME_ORDER, "mean_travel_time_s"].values
c2t = cap2.loc[TIME_ORDER, "mean_travel_time_s"].values
c1q = cap1.loc[TIME_ORDER, "mean_queue_vehicles"].values
c2q = cap2.loc[TIME_ORDER, "mean_queue_vehicles"].values
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(x - w / 2, c1t, w, label="1 lane")
axes[0].bar(x + w / 2, c2t, w, label="2 lanes")
axes[0].set_xticks(x, TIME_ORDER, rotation=45)
axes[0].set_ylabel("Mean travel time (s)")
axes[0].set_title("Travel time")
axes[0].legend()
axes[1].bar(x - w / 2, c1q, w, label="1 lane")
axes[1].bar(x + w / 2, c2q, w, label="2 lanes")
axes[1].set_xticks(x, TIME_ORDER, rotation=45)
axes[1].set_ylabel("Mean queue (vehicles)")
axes[1].set_title("Mean queue")
fig.suptitle("Figure 6 — Capacity effect (same demand, 1 vs 2 lanes)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_capacity_effect.png")
plt.close(fig)

print("Results synthesis complete:")
print(f"  master_experiments.csv : {len(master)} runs")
print(f"  tables 1-4 + summary_stats.json")
print(f"  figures: {sorted(p.name for p in FIG_DIR.glob('*.png'))}")
print(f"  demand-queue r = {summary['volume_demand_queue_pearson_r']:.3f}")
print(f"  demand-travel r = {summary['volume_demand_travel_pearson_r']:.3f}")
print(f"  capacity travel reduction: mean "
      f"{summary['capacity_travel_reduction_s']['mean']:.1f}s "
      f"(range {summary['capacity_travel_reduction_s']['min']:.1f}–"
      f"{summary['capacity_travel_reduction_s']['max']:.1f}s)")
print(f"  capacity queue reduction: mean "
      f"{summary['capacity_mean_queue_reduction_veh']['mean']:.1f} veh")
print(f"  heavy-mix travel effect: mean "
      f"{summary['heavy_composition_effect_mean']['travel_time_s']:+.1f}s")
