# TrafficPulse

Predictive traffic-demand modeling + microscopic simulation for a representative urban intersection.

Pipeline: **TIRTL observations → XGBoost short-term demand forecast → SUMO simulation (72 experiments) → frozen-results Streamlit dashboard.**

## Clone (requires Git LFS)

`*.csv` and `*.joblib` (~3 GB) are stored with Git LFS. Install it
**once per machine before cloning**, or you will get 1 KB pointer stubs
instead of data (and `pandas`/`joblib` will fail to load them):

```bash
sudo pacman -S git-lfs   # or: sudo apt install git-lfs / brew install git-lfs
git lfs install          # once per machine
git clone https://github.com/shreyas20097/trafficpulse.git
```

(Debian/Ubuntu need `git lfs install` after the apt install too;
GitHub Desktop already bundles LFS.)

Lightweight clone — skip the 1.2 GB random forest + 1.6 GB raw CSVs
if you only need the dashboard:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/shreyas20097/trafficpulse.git
cd trafficpulse
git lfs pull -I "artifacts/tirtl_training/xgboost.joblib,artifacts/results/*,artifacts/figures/*"
```

Note for the repo owner: LFS downloads count against the owner's
bandwidth quota (10 GiB/month on the Free plan ≈ 3 full clones),
so prefer the lightweight pattern when sharing.

## Quickstart

```bash
# Run everything from the repo root. Paths resolve via __file__,
# but SUMO configs use relative net-file paths, so cwd=root is required.
cd /path/to/de

# ML environment (training, evaluation, dashboard)
.venv/bin/python train_tirtl_overnight.py
.venv/bin/python evaluate_trafficpulse.py
.venv/bin/streamlit run streamlit_app.py

# SUMO environment (simulations, true-state capture)
.sumo-venv/bin/python run_volume_experiments.py --help
.sumo-venv/bin/python run_composition_experiments.py --help
.sumo-venv/bin/python run_capacity_experiments.py --help
.sumo-venv/bin/python capture_true_state.py
.sumo-venv/bin/python render_true_state.py
```

Environments: `env/env_main.txt` (ML pip freeze), `env/env_sumo.txt` (eclipse-sumo 1.27.1).
Prebuilt venvs `.venv/` and `.sumo-venv/` are git-ignored.

## Pipeline (file → purpose)

| Stage | Files |
|---|---|
| 1. Preprocess | `preprocess_tirtl.py`: `data/raw_tirtl_jan2026/TIRTLDATA_*.csv` → `artifacts/tirtl_traffic_15min.csv`, `artifacts/tirtl_features.csv` |
| 2. Train | `train_tirtl_overnight.py`: features → `artifacts/tirtl_training/` (`xgboost.joblib`, `random_forest.joblib`, `model_comparison.csv`, `test_predictions.csv`, `training_summary.json`) |
| 3. Forecast → SUMO | `trafficpulse_dynamic.py` (the controller: loads `xgboost.joblib`, writes `sumo/trafficpulse/dynamic/demand/*.rou.xml`, runs SUMO via `.sumo-venv/bin/sumo`) |
| 4. Evaluate | `evaluate_trafficpulse.py` → `artifacts/evaluation/trafficpulse_evaluation.{csv,json}` (8 periods on 2026-01-17) |
| 5. Experiments | `run_volume_experiments.py` (40 runs) → `sumo/trafficpulse/volume/`, `artifacts/experiments/volume/`; `run_composition_experiments.py` (16) → `…/composition/`; `run_capacity_experiments.py` (16) → `…/capacity/` (1-lane vs 2-lane) |
| 6. Synthesize | `synthesize_results.py` → `artifacts/results/` (`master_experiments.csv`, `summary_stats.json`, `table1-4*.csv`, `figures/fig1-6.png`) |
| 7. Visualize | `capture_true_state.py` (libsumo seed 42 → `artifacts/figures/trafficpulse/true_state/states.json`) + `render_true_state.py` (→ `true_*.png` + `manifest.json`); `build_trafficpulse_sumo.py` + `run_trafficpulse.py` = legacy baseline only |
| 8. Dashboard | `streamlit_app.py`: **frozen front end**, reads `artifacts/results/` + `true_state/` only, launches no simulations |

Helpers: `scripts/hi.sh` (patch + audit the dynamic controller; backups → `archive/trafficpulse_dynamic_history/`), `scripts/fix_trafficpulse_sumo.sh` (regenerate baseline network/demand files).

## Layout

```
streamlit_app.py  trafficpulse_dynamic.py  train_*.py  preprocess_*.py
run_*_experiments.py  evaluate_*  synthesize_*  capture/render_true_state.py
build_trafficpulse_sumo.py  run_trafficpulse.py   # legacy baseline
data/raw_tirtl_jan2026/   # 31 daily TIRTL CSVs (Jan 2026)
data/tirtl_sites.csv      # site metadata (currently unused by code)
artifacts/                # generated: features, training, evaluation, experiments, results, figures
sumo/trafficpulse/        # network/ dynamic/ volume/ composition/ capacity/ output/
scripts/  env/  logs/  archive/trafficpulse_dynamic_history/
```

## Results (frozen, see `artifacts/results/summary_stats.json`)

- XGBoost test: MAE 38.9, R² 0.985 (beats Random Forest and persistence).
- End-to-end forecast error ≈ 0.9% mean pct error.
- 72/72 experiment runs completed (40 volume + 16 composition + 16 capacity).
- Demand↔queue Pearson r ≈ 0.99; 2-lane capacity cuts travel ≈ 68 s mean; heavy mix adds ≈ 6 s travel.

## Notes for agents / developers

- **Run from root.** All Python entry points resolve paths from `Path(__file__).parent`.
- **Four source streams, not four arms.** Controller models independent TIRTL streams N/S/E/W (sites 133/126/147/195); the SUMO intersection is representative, not a replica of one real junction.
- **No synthetic graphics.** Dashboard figures under `true_state/` are real libsumo `(x, y, type)` states at t=1500 s drain phase. The old `render_trafficpulse_graphics.py` was retired.
- **Large binaries tracked:** `artifacts/tirtl_training/random_forest.joblib` (~1.2 GB), `xgboost.joblib`, plus `tirtl_features.csv` / `tirtl_traffic_15min.csv` (~800k rows each).
- **Old code:** `archive/trafficpulse_dynamic_history/` holds 14 pre-fix `trafficpulse_dynamic.py` variants (git history preserved via `git mv`).
