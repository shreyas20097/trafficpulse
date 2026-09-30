from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT / ARTIFACT PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT / "artifacts" / "results"

MASTER_FILE = RESULTS_DIR / "master_experiments.csv"
ML_FILE = RESULTS_DIR / "table1_ml_performance.csv"
FORECAST_FILE = RESULTS_DIR / "table2_forecast_validation.csv"
VOLUME_FILE = RESULTS_DIR / "table3_volume.csv"
INFRA_FILE = RESULTS_DIR / "table4_infra_composition.csv"
SUMMARY_FILE = RESULTS_DIR / "summary_stats.json"
FIGURES_DIR = RESULTS_DIR / "figures"
TP_FIG_DIR = PROJECT / "artifacts" / "figures" / "trafficpulse"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="TrafficPulse",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("TrafficPulse")
st.caption(
    "Predictive traffic-demand modeling and microscopic simulation "
    "for a representative urban intersection"
)

st.markdown(
    """
TrafficPulse combines historical TIRTL traffic observations, XGBoost
short-term demand forecasting, and SUMO microscopic simulation.

The dashboard is a **frozen-results front end**: it reads the completed
ML evaluation and 72 SUMO experiment results without launching new
simulations.
"""
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def fmt_num(value, digits: int = 1) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "—"


def pct(value, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except Exception:
        return "—"


@st.cache_data
def load_figure_bytes(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path.read_bytes()


# ============================================================
# LOAD
# ============================================================

try:
    master = load_csv(MASTER_FILE)
    ml = load_csv(ML_FILE)
    forecast = load_csv(FORECAST_FILE)
    volume = load_csv(VOLUME_FILE)
    infra = load_csv(INFRA_FILE)
    summary = load_json(SUMMARY_FILE)
except FileNotFoundError as exc:
    st.error(
        f"Required TrafficPulse result artifact is missing:\n\n{exc}"
    )
    st.info(
        "Run synthesize_results.py first so artifacts/results/ "
        "contains the frozen result package."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard")

page = st.sidebar.radio(
    "View",
    [
        "Guided demo",
        "Overview",
        "Forecast validation",
        "Demand sensitivity",
        "Composition",
        "Capacity",
        "Raw results",
    ],
)

st.sidebar.divider()

available_times = sorted(
    forecast["time"].astype(str).unique()
)

selected_time = st.sidebar.selectbox(
    "Evaluation period",
    available_times,
    index=(
        available_times.index("17:15")
        if "17:15" in available_times
        else 0
    ),
)

st.sidebar.caption(
    "All displayed values come from the completed experiment artifacts."
)


# ============================================================
# TOP-LEVEL STATUS
# ============================================================

# ============================================================
# GUIDED DEMO — one sequential walkthrough of the whole project
# ============================================================

DEMO_STEPS = [
    "Problem",
    "Data",
    "Features & ML",
    "Forecast validation",
    "Calibration → SUMO",
    "Demand experiment",
    "Composition experiment",
    "Capacity experiment",
    "Conclusion",
]

if "demo_step" not in st.session_state:
    st.session_state.demo_step = 0

if page == "Guided demo":
    step = st.session_state.demo_step
    st.progress((step + 1) / len(DEMO_STEPS))
    st.caption(
        f"Step {step + 1} of {len(DEMO_STEPS)} — {DEMO_STEPS[step]}"
    )

    go = st.sidebar.selectbox(
        "Jump to step",
        range(len(DEMO_STEPS)),
        index=step,
        format_func=lambda i: f"{i + 1}. {DEMO_STEPS[i]}",
    )
    if go != step:
        st.session_state.demo_step = go
        st.rerun()

    def tp_fig(name: str, caption: str = ""):
        p = TP_FIG_DIR / name
        if p.exists():
            st.image(
                load_figure_bytes(p),
                caption=caption or None,
                width="stretch",
            )

    # ---- 0 Problem ----
    if step == 0:
        st.subheader("1 · Problem")
        st.markdown(
            """
**User:** traffic planner / simulation analyst.

**Pain points:** static manual demand inputs, shifting traffic
conditions, varying vehicle mix, and no clean way to compare
infrastructure scenarios.

**Design need:** representative + dynamic + data-driven demand.

> TrafficPulse predicts short-term traffic demand from TIRTL data,
> converts that prediction into a representative microscopic SUMO
> intersection, and uses controlled experiments to study how demand,
> vehicle composition, and road capacity affect simulated performance.
"""
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("SUMO runs", "72")
        c2.metric("XGBoost test R²", fmt_num(summary.get("xgboost_test_r2"), 4))
        c3.metric("Forecast mean error", pct(summary.get("mean_source_pct_error"), 2))
        tp_fig(
            "trafficpulse_intersection_hero.png",
            "Four-arm signalized intersection, top-down.",
        )

    # ---- 1 Data ----
    elif step == 1:
        st.subheader("2 · Data — Victorian TIRTL, January 2026")
        st.markdown(
            """
- 15-minute temporal resolution, E/N/S/W heading labels
- 15 original vehicle classes grouped into 5 simulation categories:
  motorcycle, short, short-towing, rigid truck/bus, heavy articulated
- Hundreds of monitored sites; cleaned subset feeds the pipeline
"""
        )
        row0 = forecast.iloc[0] if not forecast.empty else None
        c1, c2, c3 = st.columns(3)
        c1.metric("Evaluation periods", f"{forecast['time'].nunique()}")
        c2.metric("Master experiment rows", f"{len(master)}")
        c3.metric("Figures", "6 + intersection set")

    # ---- 2 Features & ML ----
    elif step == 2:
        st.subheader("3 · Features & ML — 327-feature input, 3 models")
        st.markdown(
            """
Temporal features, current class counts, total demand, lagged demand,
rolling statistics, encoded site + heading. Target: next-15-minute
motorized demand. Time-aware split — no future leakage into training.
"""
        )
        st.dataframe(ml, width="stretch", hide_index=True)
        st.success("Lowest error: XGBoost (MAE 38.9, R² 0.985); used downstream.")
        p = FIGURES_DIR / "fig1_model_comparison.png"
        if p.exists():
            st.image(load_figure_bytes(p), caption="Model comparison",
                     width="stretch")

    # ---- 3 Forecast validation ----
    elif step == 3:
        st.subheader("4 · Forecast validation — 8 representative periods")
        st.markdown(
            "End-to-end check, separate from the ML held-out test: "
            "TIRTL → features → XGBoost → calibration → SUMO."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Source mean % error",
                  pct(summary["end_to_end"]["mean_source_pct_error"], 2))
        c2.metric("Representative MAE",
                  fmt_num(summary["end_to_end"]["representative_demand_mae"], 2))
        c3.metric("Representative mean % error",
                  pct(summary["end_to_end"]["mean_representative_pct_error"], 2))
        st.dataframe(forecast, width="stretch", hide_index=True)

    # ---- 4 Calibration → SUMO ----
    elif step == 4:
        st.subheader("5 · Calibration → SUMO intersection")
        st.markdown(
            """
ML output is network-wide demand, **not** an intersection count.
Calibration converts it to representative 15-minute demand, then
directional shares (N 20.6 / S 20.3 / E 29.3 / W 29.7 %, assumed) and
turn shares (left 20 / straight 70 / right 10 %, assumed) assign
every vehicle.

Network: 4 arms, signalized, 1 lane/approach, 50 km/h, 66 s cycle
(30 s green + 3 s yellow), SUMO 1.27.1, fixed seed 42.
"""
        )
        tp_fig("01_prototype_sparse.png", "Baseline layout, sparse demand")
        tp_fig("02_forecast_to_sumo.png", "Demand inflow to the network")

    # ---- 5 Demand ----
    elif step == 5:
        st.subheader("6 · Demand experiment — 8 × 5 = 40 runs")
        st.markdown(
            "Only total demand changes (80 % / predicted / fixed-1050 / "
            "120 % / 140 %). Network, signal, shares, composition, seed frozen."
        )
        c1, c2 = st.columns(2)
        c1.metric("Demand → queue r",
                  f"{summary.get('demand_queue_pearson_r', 0):.3f}")
        c2.metric("Demand → travel-time r",
                  f"{summary.get('volume_demand_travel_pearson_r', 0):.3f}")
        i1, i2 = st.columns(2)
        with i1:
            tp_fig("03_demand_low.png", "Low")
        with i2:
            tp_fig("03_demand_base.png", "Base")
        i3, i4 = st.columns(2)
        with i3:
            tp_fig("03_demand_high.png", "High")
        with i4:
            tp_fig("03_demand_peak.png", "Peak")
        st.dataframe(
            volume.sort_values(["time", "demand_veh_15min"])[
                ["time", "scenario", "demand_veh_15min",
                 "mean_travel_time_s", "mean_waiting_time_s",
                 "mean_queue_vehicles", "max_queue_vehicles"]
            ],
            width="stretch", hide_index=True,
        )

    # ---- 6 Composition ----
    elif step == 6:
        st.subheader("7 · Composition experiment — 8 × 2 = 16 runs")
        st.markdown(
            "Demand, network, signal fixed. Normal audited TIRTL mix vs "
            "heavy mix — same total vehicles, cyan dots turn amber."
        )
        h = summary.get("heavy_composition_effect_mean", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Travel effect", f"+{h.get('travel_time_s', 0):.1f} s")
        c2.metric("Waiting effect", f"+{h.get('waiting_time_s', 0):.1f} s")
        c3.metric("Queue effect", f"+{h.get('mean_queue_veh', 0):.1f} veh")
        i1, i2 = st.columns(2)
        with i1:
            tp_fig("04a_composition_all_cars.png", "All cars (reference)")
        with i2:
            tp_fig("04b_composition_heavy_mix.png", "Heavy mix, same total")
        st.warning("Note: the 20:00 period is an outlier in the composition results.")

    # ---- 7 Capacity ----
    elif step == 7:
        st.subheader("8 · Capacity experiment — 8 × 2 = 16 runs")
        st.markdown(
            "Demand and composition frozen. One lane vs two lanes per approach."
        )
        c1, c2 = st.columns(2)
        c1.metric("Mean travel reduction",
                  f"{summary.get('mean_two_lane_travel_reduction_s', 0):.1f} s")
        c2.metric("Mean queue reduction",
                  f"{summary['capacity_mean_queue_reduction_veh']['mean']:.1f} veh")
        st.caption(
            f"Range: travel {summary['capacity_travel_reduction_s']['min']:.1f}–"
            f"{summary['capacity_travel_reduction_s']['max']:.1f} s · queue "
            f"{summary['capacity_mean_queue_reduction_veh']['min']:.1f}–"
            f"{summary['capacity_mean_queue_reduction_veh']['max']:.1f} veh"
        )
        i1, i2 = st.columns(2)
        with i1:
            tp_fig("05a_capacity_one_lane.png", "One lane / approach")
        with i2:
            tp_fig("05b_capacity_two_lanes.png", "Two lanes / approach")

    # ---- 8 Conclusion ----
    elif step == 8:
        st.subheader("9 · Conclusion — back to the same intersection")
        tp_fig("06_conclusion_clean.png", "Baseline layout")
        st.markdown(
            """
**Scope:** hypothetical intersection; TIRTL streams are independent
sensors, not one real junction; turns and directional shares are
assumed values; SUMO numbers are **comparative simulation output**,
not measured real-world performance.

**Reproducibility:** fixed seed 42 · frozen network/signal · 72 runs
(40 + 16 + 16) · frozen CSV/JSON artifacts · this dashboard reads
precomputed results only.
"""
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("XGBoost MAE", fmt_num(summary.get("xgboost_test_mae"), 1))
        c2.metric("Queue r", f"{summary.get('demand_queue_pearson_r', 0):.3f}")
        c3.metric("2-lane Δtravel",
                  f"−{summary.get('mean_two_lane_travel_reduction_s', 0):.1f} s")
        c4.metric("Heavy Δtravel",
                  f"+{summary.get('mean_heavy_travel_delta_s', 0):.1f} s")

    st.divider()
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("◀ Back", disabled=(step == 0), width="stretch"):
            st.session_state.demo_step = max(0, step - 1)
            st.rerun()
    with b3:
        if st.button("Next ▶", disabled=(step == len(DEMO_STEPS) - 1),
                     width="stretch"):
            st.session_state.demo_step = min(len(DEMO_STEPS) - 1, step + 1)
            st.rerun()

elif page == "Overview":
    st.subheader("Project status")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "XGBoost test MAE",
        fmt_num(summary.get("xgboost_test_mae"), 2),
    )
    c2.metric(
        "XGBoost test R²",
        fmt_num(summary.get("xgboost_test_r2"), 4),
    )
    c3.metric(
        "Forecast mean error",
        pct(summary.get("mean_source_pct_error"), 2),
    )
    c4.metric(
        "SUMO experiments",
        "72",
    )

    st.divider()

    st.subheader("TrafficPulse pipeline")

    st.code(
        """TIRTL traffic data
        ↓
Feature engineering
        ↓
Persistence / Random Forest / XGBoost
        ↓
Next-15-minute demand forecast
        ↓
Representative intersection calibration
        ↓
SUMO microscopic simulation
        ↓
Demand / composition / capacity experiments
        ↓
Traffic metrics""",
        language="text",
    )

    st.subheader("Headline results")

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Demand → queue",
        f"r = {summary.get('demand_queue_pearson_r', 0):.3f}",
        help="Pearson correlation from the controlled volume experiment.",
    )

    h2.metric(
        "2-lane travel reduction",
        f"{summary.get('mean_two_lane_travel_reduction_s', 0):.1f} s",
        help="Mean reduction versus the corresponding 1-lane case.",
    )

    h3.metric(
        "Heavy-mix travel effect",
        f"+{summary.get('mean_heavy_travel_delta_s', 0):.1f} s",
        help="Heavy composition minus normal composition at equal demand.",
    )

    st.subheader("Representative-period forecast")

    row = forecast[
        forecast["time"].astype(str) == selected_time
    ]

    if not row.empty:
        r = row.iloc[0]

        a, b, c, d = st.columns(4)

        a.metric(
            "Observed next 15 min",
            f"{float(r['actual_demand']):,.0f}",
        )
        b.metric(
            "TrafficPulse prediction",
            f"{float(r['predicted_demand']):,.0f}",
        )
        c.metric(
            "Absolute error",
            f"{float(r['absolute_error']):.1f}",
        )
        d.metric(
            "Percentage error",
            pct(r["percentage_error"], 2),
        )

    st.subheader("Key figures")

    figure_names = [
        ("fig1_model_comparison.png", "Model comparison"),
        ("fig2_actual_vs_predicted.png", "Actual vs predicted demand"),
        ("fig3_demand_vs_travel.png", "Demand vs travel time"),
        ("fig4_demand_vs_queue.png", "Demand vs queue"),
        ("fig5_composition_effect.png", "Vehicle composition effect"),
        ("fig6_capacity_effect.png", "Lane capacity effect"),
    ]

    for filename, caption in figure_names:
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(
                load_figure_bytes(path),
                caption=caption,
                width="stretch",
            )


# ============================================================
# FORECAST VALIDATION
# ============================================================

elif page == "Forecast validation":
    st.subheader("Forecast validation")

    st.write(
        "Eight representative January 2026 periods are compared "
        "against the observed next-15-minute TIRTL demand."
    )

    display = forecast.copy()

    numeric_cols = [
        c for c in display.columns
        if c not in {"date", "time"}
    ]

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )

    path = FIGURES_DIR / "fig2_actual_vs_predicted.png"
    if path.exists():
        st.image(
            load_figure_bytes(path),
            caption="Actual vs predicted representative demand",
            width="stretch",
        )


# ============================================================
# DEMAND SENSITIVITY
# ============================================================

elif page == "Demand sensitivity":
    st.subheader("Demand sensitivity")

    st.write(
        """
The demand experiment changes only total demand. Network geometry,
signal timing, direction shares, turn shares, vehicle composition, and
random seed remain fixed.
"""
    )

    times = sorted(
        volume["time"].astype(str).unique()
    )

    time_filter = st.selectbox(
        "Period",
        times,
        index=(
            times.index(selected_time)
            if selected_time in times
            else 0
        ),
    )

    selected = volume[
        volume["time"].astype(str) == time_filter
    ].copy()

    selected = selected.sort_values(
        "demand_veh_15min"
    )

    st.dataframe(
        selected[
            [
                "time",
                "scenario",
                "demand_veh_15min",
                "mean_travel_time_s",
                "mean_waiting_time_s",
                "mean_time_loss_s",
                "mean_queue_vehicles",
                "max_queue_vehicles",
                "vehicles_arrived",
                "completion_rate_pct",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Demand → congestion")

    c1, c2 = st.columns(2)

    with c1:
        st.image(
            load_figure_bytes(
                FIGURES_DIR / "fig3_demand_vs_travel.png"
            ),
            caption="Demand vs mean travel time",
            width="stretch",
        )

    with c2:
        st.image(
            load_figure_bytes(
                FIGURES_DIR / "fig4_demand_vs_queue.png"
            ),
            caption="Demand vs mean queue",
            width="stretch",
        )


# ============================================================
# COMPOSITION
# ============================================================

elif page == "Composition":
    st.subheader("Vehicle-composition sensitivity")

    st.write(
        """
Total demand is held constant while vehicle composition changes from
the audited TIRTL hourly mix to the controlled heavy-vehicle mix.
"""
    )

    times = sorted(
        infra["time"].astype(str).unique()
    )

    comp = infra[
        (infra["time"].astype(str) == selected_time)
        & (infra["experiment"].astype(str) == "composition")
    ].copy()

    if "scenario" in comp.columns:
        comp = comp[
            comp["scenario"].isin(
                ["normal", "heavy"]
            )
        ]

    st.dataframe(
        comp[
            [
                c
                for c in [
                    "time",
                    "experiment",
                    "scenario",
                    "demand_veh_15min",
                    "short_share",
                    "towing_share",
                    "rigid_share",
                    "heavy_share",
                    "motorcycle_share",
                    "mean_travel_time_s",
                    "mean_waiting_time_s",
                    "mean_queue_vehicles",
                    "max_queue_vehicles",
                ]
                if c in comp.columns
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.image(
        load_figure_bytes(
            FIGURES_DIR / "fig5_composition_effect.png"
        ),
        caption="Normal vs heavy vehicle composition",
        width="stretch",
    )

    st.warning(
        "The 20:00 composition result is an identified outlier in the "
        "completed experiment and should be discussed rather than hidden."
    )


# ============================================================
# CAPACITY
# ============================================================

elif page == "Capacity":
    st.subheader("Road-capacity sensitivity")

    st.write(
        """
TrafficPulse demand and TIRTL vehicle composition are held constant.
Only the number of lanes per approach changes from one to two.
"""
    )

    cap = infra[
        (infra["time"].astype(str) == selected_time)
        & (infra["experiment"].astype(str) == "capacity")
    ].copy()

    if "lanes_per_approach" in cap.columns:
        cap = cap.sort_values(
            "lanes_per_approach"
        )

        st.dataframe(
            cap[
                [
                    c
                    for c in [
                        "time",
                        "lanes_per_approach",
                        "demand_veh_15min",
                        "mean_travel_time_s",
                        "mean_waiting_time_s",
                        "mean_time_loss_s",
                        "mean_queue_vehicles",
                        "max_queue_vehicles",
                        "completion_rate_pct",
                    ]
                    if c in cap.columns
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    reduction = None

    if (
        "lanes_per_approach" in cap.columns
        and len(cap) >= 2
    ):
        one = cap[
            cap["lanes_per_approach"] == 1
        ]

        two = cap[
            cap["lanes_per_approach"] == 2
        ]

        if not one.empty and not two.empty:
            reduction = (
                float(one.iloc[0]["mean_travel_time_s"])
                - float(two.iloc[0]["mean_travel_time_s"])
            )

    if reduction is not None:
        st.metric(
            "Travel-time reduction from 2 lanes",
            f"{reduction:.1f} s/vehicle",
        )

    st.image(
        load_figure_bytes(
            FIGURES_DIR / "fig6_capacity_effect.png"
        ),
        caption="One lane vs two lanes per approach",
        width="stretch",
    )


# ============================================================
# RAW RESULTS
# ============================================================

elif page == "Raw results":
    st.subheader("Frozen experiment dataset")

    experiment_filter = st.selectbox(
        "Experiment",
        ["All"] + sorted(
            master["experiment"]
            .astype(str)
            .unique()
            .tolist()
        ),
    )

    raw = master.copy()

    if experiment_filter != "All":
        raw = raw[
            raw["experiment"].astype(str)
            == experiment_filter
        ]

    st.dataframe(
        raw,
        width="stretch",
        height=600,
        hide_index=True,
    )

    csv_data = raw.to_csv(index=False).encode()

    st.download_button(
        "Download filtered CSV",
        data=csv_data,
        file_name="trafficpulse_filtered_results.csv",
        mime="text/csv",
    )

    st.subheader("ML performance")

    st.dataframe(
        ml,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Summary JSON")

    st.json(summary)
