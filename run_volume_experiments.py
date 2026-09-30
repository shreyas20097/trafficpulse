#!/usr/bin/env python3
"""
TrafficPulse controlled volume-sensitivity experiment runner.

Phase 3-5 of the TrafficPulse roadmap:
  What happens to intersection performance when traffic demand changes?

Design (one variable at a time):
  - Independent variable: total demand only.
  - Frozen: network (1 lane/approach, 50 km/h), signal timing (66 s cycle,
    30 s green + 3 s yellow per opposing phase), directional shares,
    turn shares (20/70/10), hourly TIRTL vehicle composition, seed.
  - For each of the 8 representative periods, 5 scenarios:
      low        = round(predicted * 0.8)
      predicted  = TrafficPulse XGBoost demand (from evaluation CSV)
      baseline   = fixed 1050 reference
      high       = round(predicted * 1.2)
      very_high  = round(predicted * 1.4)

Metrics per scenario (trip-level from tripinfo + network queue via libsumo):
  demand, throughput, completion, mean travel/waiting/time-loss,
  mean queue, max queue.

Outputs:
  artifacts/experiments/volume/volume_scenarios.csv
  artifacts/experiments/volume/volume_scenarios.json
  sumo/trafficpulse/volume/<tag>.rou.xml / .sumocfg / output/*.tripinfo.xml

Run with the SUMO venv (needs libsumo, no pandas/sklearn required):
  .sumo-venv/bin/python run_volume_experiments.py
  .sumo-venv/bin/python run_volume_experiments.py --dry-run
  .sumo-venv/bin/python run_volume_experiments.py --times 17:15 --scenarios predicted,high
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import libsumo
except ImportError as exc:
    raise SystemExit(
        "libsumo is required. Run with: .sumo-venv/bin/python "
        f"run_volume_experiments.py ({exc})"
    )


PROJECT = Path(__file__).resolve().parent
EVAL_CSV = PROJECT / "artifacts/evaluation/trafficpulse_evaluation.csv"
COMPOSITION_FILE = PROJECT / "artifacts/tirtl_vehicle_composition_by_hour.csv"
NETWORK = PROJECT / "sumo/trafficpulse/network/intersection.net.xml"

VOLUME_DIR = PROJECT / "sumo/trafficpulse/volume"
VOLUME_DEMAND_DIR = VOLUME_DIR / "demand"
VOLUME_OUTPUT_DIR = VOLUME_DIR / "output"
RESULT_DIR = PROJECT / "artifacts/experiments/volume"

# Frozen baseline configuration (do not change during volume experiment).
DEMAND_DURATION = 900
SIM_END = 3600
SEED = 42
BASELINE_DEMAND = 1050

DIRECTION_SHARE = {
    "N": 0.2061,
    "S": 0.2034,
    "E": 0.2934,
    "W": 0.2971,
}

TURN_SHARE = {
    "left": 0.20,
    "straight": 0.70,
    "right": 0.10,
}

MOVEMENTS = {
    "N": {
        "left": "N_C C_E",
        "straight": "N_C C_S",
        "right": "N_C C_W",
    },
    "S": {
        "left": "S_C C_W",
        "straight": "S_C C_N",
        "right": "S_C C_E",
    },
    "E": {
        "left": "E_C C_N",
        "straight": "E_C C_W",
        "right": "E_C C_S",
    },
    "W": {
        "left": "W_C C_S",
        "straight": "W_C C_E",
        "right": "W_C C_N",
    },
}

INCOMING_EDGES = ["N_C", "S_C", "E_C", "W_C"]

# TIRTL audit column -> SUMO vType.
COMPOSITION_MAP = {
    "short_vehicle": "short",
    "short_vehicle_towing": "towing",
    "rigid_truck_bus": "rigid",
    "heavy_articulated": "heavy",
    "motorcycle": "motorcycle",
}

VTYPES_XML = """<routes>

    <vType
        id="short"
        accel="2.6"
        decel="4.5"
        sigma="0.5"
        length="4.5"
        minGap="2.5"
        maxSpeed="13.89"
        guiShape="passenger"
        color="8,185,242"
    />

    <vType
        id="towing"
        accel="2.2"
        decel="4.5"
        sigma="0.5"
        length="7.0"
        minGap="3.0"
        maxSpeed="13.89"
        guiShape="passenger"
        color="0,224,255"
    />

    <vType
        id="rigid"
        accel="1.8"
        decel="4.0"
        sigma="0.5"
        length="10.0"
        minGap="3.0"
        maxSpeed="13.89"
        guiShape="truck"
        color="8,185,242"
    />

    <vType
        id="heavy"
        accel="1.4"
        decel="3.5"
        sigma="0.5"
        length="18.0"
        minGap="4.0"
        maxSpeed="13.89"
        guiShape="truck"
        color="243,154,61"
    />

    <vType
        id="motorcycle"
        accel="3.0"
        decel="5.0"
        sigma="0.5"
        length="2.2"
        minGap="1.5"
        maxSpeed="13.89"
        guiShape="motorcycle"
        color="0,224,255"
    />

"""

# Scenario -> multiplier applied to the TrafficPulse predicted demand.
# "baseline" is the fixed 1050 reference (multiplier None).
SCENARIOS: dict[str, float | None] = {
    "low": 0.8,
    "predicted": 1.0,
    "baseline": None,
    "high": 1.2,
    "very_high": 1.4,
}
SCENARIO_ORDER = ["low", "predicted", "baseline", "high", "very_high"]

EVAL_DATE_DEFAULT = "2026-01-17"
EVAL_TIMES_DEFAULT = [
    "06:30",
    "08:30",
    "10:45",
    "12:15",
    "14:30",
    "17:15",
    "18:45",
    "20:00",
]


def exact_allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    keys = list(shares)
    raw = {k: total * shares[k] for k in keys}
    values = {k: math.floor(raw[k]) for k in keys}
    remaining = total - sum(values.values())
    order = sorted(keys, key=lambda k: raw[k] - values[k], reverse=True)
    for k in order[:remaining]:
        values[k] += 1
    return values


def load_predicted_demands(eval_csv: Path) -> dict[str, int]:
    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing evaluation CSV: {eval_csv}")
    out: dict[str, int] = {}
    with eval_csv.open() as f:
        for row in csv.DictReader(f):
            t = row["time"].strip()
            out[t] = int(round(float(row["predicted_representative_demand_veh_15min"])))
    return out


def load_composition(hour: int) -> dict[str, float]:
    if not COMPOSITION_FILE.exists():
        raise FileNotFoundError(f"Missing composition file: {COMPOSITION_FILE}")
    with COMPOSITION_FILE.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            # First column is the hour index (named "hour" or "").
            h_raw = row.get("hour", row.get("", "")).strip()
            if h_raw == "":
                # Fall back to first value.
                h_raw = list(row.values())[0].strip()
            if int(float(h_raw)) != int(hour):
                continue
            comp = {}
            for src, sumo_type in COMPOSITION_MAP.items():
                comp[sumo_type] = float(row[f"{src}_share"])
            total = sum(comp.values())
            if total <= 0:
                raise RuntimeError("Vehicle composition sums to zero.")
            return {k: v / total for k, v in comp.items()}
    raise ValueError(f"No vehicle composition for hour {hour}")


def build_routes_xml(total: int, composition: dict[str, float]) -> tuple[str, dict]:
    direction_counts = exact_allocate(total, DIRECTION_SHARE)
    assert sum(direction_counts.values()) == total

    lines: list[str] = []
    for direction in ("N", "S", "E", "W"):
        for turn in ("left", "straight", "right"):
            lines.append(
                f'    <route id="{direction}_{turn}" '
                f'edges="{MOVEMENTS[direction][turn]}" />\n'
            )

    flow_id = 0
    for direction in ("N", "S", "E", "W"):
        movement_counts = exact_allocate(direction_counts[direction], TURN_SHARE)
        assert sum(movement_counts.values()) == direction_counts[direction]
        for turn in ("left", "straight", "right"):
            veh_counts = exact_allocate(movement_counts[turn], composition)
            assert sum(veh_counts.values()) == movement_counts[turn]
            for vtype, n in veh_counts.items():
                if n <= 0:
                    continue
                lines.append(
                    f'    <flow id="f_{flow_id}" type="{vtype}" '
                    f'route="{direction}_{turn}" begin="0" '
                    f'end="{DEMAND_DURATION}" number="{n}" '
                    f'departLane="best" departSpeed="max" />\n'
                )
                flow_id += 1

    return VTYPES_XML + "".join(lines) + "</routes>\n", direction_counts


def read_tripinfo(path: Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"SUMO did not create TripInfo: {path}")
    root = ET.parse(path).getroot()
    trips = []
    for el in root.findall("tripinfo"):
        arrival = float(el.attrib.get("arrival", "-1"))
        duration = float(el.attrib.get("duration", "-1"))
        trips.append({
            "arrival": arrival,
            "duration": duration,
            "waitingTime": float(el.attrib.get("waitingTime", "nan")),
            "timeLoss": float(el.attrib.get("timeLoss", "nan")),
            "finished": arrival >= 0 and duration >= 0,
        })
    return trips


def run_sumo(cfg: Path, tripinfo: Path) -> dict:
    if tripinfo.exists():
        tripinfo.unlink()
    libsumo.start(["sumo", "-c", str(cfg), "--seed", str(SEED), "--no-warnings", "true"])
    queue_samples: list[int] = []
    sim_end = 0.0
    try:
        while libsumo.simulation.getTime() < SIM_END:
            libsumo.simulationStep()
            sim_end = libsumo.simulation.getTime()
            q = sum(
                libsumo.edge.getLastStepHaltingNumber(e) for e in INCOMING_EDGES
            )
            queue_samples.append(q)
    finally:
        libsumo.close()

    trips = read_tripinfo(tripinfo)
    finished = [t for t in trips if t["finished"]]
    travel = [t["duration"] for t in finished if t["duration"] >= 0]
    waiting = [t["waitingTime"] for t in finished if t["waitingTime"] >= 0]
    loss = [t["timeLoss"] for t in finished if t["timeLoss"] >= 0]
    total = len(trips)
    arrived = len(finished)
    return {
        "simulation_end_s": sim_end,
        "vehicles_reported": total,
        "vehicles_arrived": arrived,
        "vehicles_unfinished": total - arrived,
        "completion_rate_pct": arrived / total * 100 if total else 0.0,
        "mean_travel_time_s": statistics.mean(travel) if travel else 0.0,
        "mean_waiting_time_s": statistics.mean(waiting) if waiting else 0.0,
        "mean_time_loss_s": statistics.mean(loss) if loss else 0.0,
        "mean_queue_vehicles": statistics.mean(queue_samples) if queue_samples else 0.0,
        "max_queue_vehicles": max(queue_samples) if queue_samples else 0,
    }


def scenario_demand(scenario: str, predicted: int) -> int:
    if scenario == "baseline":
        return BASELINE_DEMAND
    factor = SCENARIOS[scenario]
    assert factor is not None
    return max(1, int(round(predicted * factor)))


def main() -> None:
    global SEED
    ap = argparse.ArgumentParser(description="TrafficPulse volume experiment runner")
    ap.add_argument("--date", default=EVAL_DATE_DEFAULT)
    ap.add_argument("--times", default=",".join(EVAL_TIMES_DEFAULT),
                    help="Comma-separated HH:MM subset")
    ap.add_argument("--scenarios", default=",".join(SCENARIO_ORDER),
                    help="Comma-separated subset of low,predicted,baseline,high,very_high")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate demand files without launching SUMO")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    SEED = args.seed

    times = [t.strip() for t in args.times.split(",") if t.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in scenarios:
        if s not in SCENARIOS:
            raise SystemExit(f"Unknown scenario '{s}'. Choose from {SCENARIO_ORDER}")

    for p in (EVAL_CSV, COMPOSITION_FILE, NETWORK):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")
    VOLUME_DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    VOLUME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    predicted_by_time = load_predicted_demands(EVAL_CSV)
    for t in times:
        if t not in predicted_by_time:
            raise SystemExit(f"Time {t} not found in {EVAL_CSV}")

    print("=" * 80)
    print("TRAFFICPULSE VOLUME-SENSITIVITY EXPERIMENTS")
    print("=" * 80)
    print(f"Date: {args.date} | Times: {', '.join(times)}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print("Frozen: 1 lane/approach, 50 km/h, 66 s cycle, 20/70/10 turns,")
    print("        directional shares, hourly TIRTL composition, seed=%d" % SEED)
    print(f"Dry run: {args.dry_run}")
    print()

    rows: list[dict] = []
    for time_str in times:
        hour = int(time_str.split(":")[0])
        predicted = predicted_by_time[time_str]
        composition = load_composition(hour)
        for scenario in scenarios:
            demand = scenario_demand(scenario, predicted)
            tag = f"vol_{args.date.replace('-', '')}_{time_str.replace(':', '')}_{scenario}"
            rou = VOLUME_DEMAND_DIR / f"{tag}.rou.xml"
            cfg = VOLUME_DIR / f"{tag}.sumocfg"
            tripinfo = VOLUME_OUTPUT_DIR / f"{tag}.tripinfo.xml"

            routes_xml, dir_counts = build_routes_xml(demand, composition)
            rou.write_text(routes_xml)
            cfg.write_text(
                f"""<configuration>
    <input>
        <net-file value="network/intersection.net.xml"/>
        <route-files value="../volume/demand/{rou.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIM_END}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
    </processing>
    <output>
        <tripinfo-output value="../volume/output/{tripinfo.name}"/>
        <tripinfo-output.write-unfinished value="true"/>
    </output>
    <random_number>
        <seed value="{SEED}"/>
    </random_number>
</configuration>
"""
            )
            # NOTE: route-files / tripinfo paths above are relative to the
            # sumocfg location (sumo/trafficpulse/volume/). The network path
            # resolves to sumo/trafficpulse/network/intersection.net.xml only
            # if the config is executed with cwd=sumo/trafficpulse/volume, but
            # libsumo resolves relative to cwd. To avoid ambiguity, rewrite
            # with absolute paths before running.
            cfg.write_text(
                f"""<configuration>
    <input>
        <net-file value="{NETWORK.resolve()}"/>
        <route-files value="{rou.resolve()}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIM_END}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
    </processing>
    <output>
        <tripinfo-output value="{tripinfo.resolve()}"/>
        <tripinfo-output.write-unfinished value="true"/>
    </output>
    <random_number>
        <seed value="{SEED}"/>
    </random_number>
</configuration>
"""
            )

            if args.dry_run:
                print(f"  [dry-run] {time_str} {scenario:<10} demand={demand} -> {rou.name}")
                rows.append({
                    "date": args.date, "time": time_str, "hour": hour,
                    "scenario": scenario, "demand_veh_15min": demand,
                    "predicted_demand_veh_15min": predicted,
                    "dry_run": True,
                })
                continue

            print(f"  Running {time_str} {scenario:<10} demand={demand} ...",
                  end=" ", flush=True)
            metrics = run_sumo(cfg, tripinfo)
            if metrics["vehicles_reported"] != demand:
                print(f"\nWARNING: inserted {metrics['vehicles_reported']} != "
                      f"demand {demand}")
            print(f"travel={metrics['mean_travel_time_s']:.1f}s "
                  f"wait={metrics['mean_waiting_time_s']:.1f}s "
                  f"queue={metrics['mean_queue_vehicles']:.1f}/"
                  f"{metrics['max_queue_vehicles']} "
                  f"completion={metrics['completion_rate_pct']:.1f}%")
            rows.append({
                "date": args.date,
                "time": time_str,
                "hour": hour,
                "scenario": scenario,
                "demand_veh_15min": demand,
                "predicted_demand_veh_15min": predicted,
                "direction_N": dir_counts["N"],
                "direction_S": dir_counts["S"],
                "direction_E": dir_counts["E"],
                "direction_W": dir_counts["W"],
                **metrics,
                "route_file": str(rou),
                "config_file": str(cfg),
                "tripinfo_file": str(tripinfo),
            })

    if args.dry_run:
        print(f"\nDry run complete: {len(rows)} demand files generated, no SUMO runs.")
        return

    out_csv = RESULT_DIR / "volume_scenarios.csv"
    out_json = RESULT_DIR / "volume_scenarios.json"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with out_json.open("w") as f:
        json.dump({
            "date": args.date,
            "times": times,
            "scenarios": scenarios,
            "frozen_config": {
                "lanes_per_approach": 1,
                "speed_kmh": 50,
                "signal_cycle_s": 66,
                "green_s": 30,
                "yellow_s": 3,
                "turn_share": TURN_SHARE,
                "direction_share": DIRECTION_SHARE,
                "demand_duration_s": DEMAND_DURATION,
                "sim_end_s": SIM_END,
                "seed": SEED,
                "baseline_demand": BASELINE_DEMAND,
            },
            "scenario_definition": {
                "low": "round(predicted*0.8)",
                "predicted": "TrafficPulse XGBoost demand",
                "baseline": "fixed 1050 reference",
                "high": "round(predicted*1.2)",
                "very_high": "round(predicted*1.4)",
            },
            "runs": rows,
        }, f, indent=2)

    print()
    print("=" * 80)
    print(f"VOLUME EXPERIMENTS COMPLETE: {len(rows)} runs")
    print(f"CSV : {out_csv}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
