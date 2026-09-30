#!/usr/bin/env python3
"""
TrafficPulse controlled vehicle-composition experiment.

Independent variable:
    Vehicle composition only.

Frozen:
    Predicted TrafficPulse demand at each evaluation time
    1 lane/approach
    50 km/h
    66 s cycle
    30 s green + 3 s yellow per opposing phase
    direction shares
    turn shares = 20/70/10
    SUMO seed

Scenarios:
    normal = audited TIRTL hourly composition
    heavy  = controlled heavy-vehicle composition:
             short 75%, towing 2%, rigid 15%, heavy 7%, motorcycle 1%

Outputs:
    artifacts/experiments/composition/composition_scenarios.csv
    artifacts/experiments/composition/composition_scenarios.json
    sumo/trafficpulse/composition/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import libsumo


PROJECT = Path(__file__).resolve().parent
EVAL_CSV = PROJECT / "artifacts/evaluation/trafficpulse_evaluation.csv"
COMPOSITION_FILE = PROJECT / "artifacts/tirtl_vehicle_composition_by_hour.csv"
NETWORK = PROJECT / "sumo/trafficpulse/network/intersection.net.xml"

BASE = PROJECT / "sumo/trafficpulse/composition"
DEMAND_DIR = BASE / "demand"
OUTPUT_DIR = BASE / "output"
RESULT_DIR = PROJECT / "artifacts/experiments/composition"

DEMAND_DURATION = 900
SIM_END = 3600
SEED = 42

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

COMPOSITION_MAP = {
    "short_vehicle": "short",
    "short_vehicle_towing": "towing",
    "rigid_truck_bus": "rigid",
    "heavy_articulated": "heavy",
    "motorcycle": "motorcycle",
}

# Controlled stress case. Sum = 100%.
HEAVY_COMPOSITION = {
    "short": 0.75,
    "towing": 0.02,
    "rigid": 0.15,
    "heavy": 0.07,
    "motorcycle": 0.01,
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

DEFAULT_DATE = "2026-01-17"
DEFAULT_TIMES = [
    "06:30",
    "08:30",
    "10:45",
    "12:15",
    "14:30",
    "17:15",
    "18:45",
    "20:00",
]

SCENARIOS = ("normal", "heavy")


def exact_allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    keys = list(shares)
    share_sum = sum(shares.values())
    if not math.isclose(share_sum, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Shares must sum to 1.0, got {share_sum}")

    raw = {k: total * shares[k] for k in keys}
    values = {k: math.floor(raw[k]) for k in keys}
    remaining = total - sum(values.values())

    order = sorted(
        keys,
        key=lambda k: raw[k] - values[k],
        reverse=True,
    )
    for k in order[:remaining]:
        values[k] += 1

    assert sum(values.values()) == total
    return values


def load_predicted_demands(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(path)

    out: dict[str, int] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            time_str = row["time"].strip()
            out[time_str] = int(
                round(
                    float(
                        row["predicted_representative_demand_veh_15min"]
                    )
                )
            )
    return out


def load_normal_composition(hour: int) -> dict[str, float]:
    if not COMPOSITION_FILE.exists():
        raise FileNotFoundError(COMPOSITION_FILE)

    with COMPOSITION_FILE.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw = row.get("hour", "")
            if not raw:
                raw = list(row.values())[0]

            if int(float(raw)) != int(hour):
                continue

            comp = {}
            for src, sumo_type in COMPOSITION_MAP.items():
                comp[sumo_type] = float(row[f"{src}_share"])

            total = sum(comp.values())
            if total <= 0:
                raise RuntimeError(
                    f"Invalid normal composition for hour {hour}"
                )

            return {
                k: v / total
                for k, v in comp.items()
            }

    raise ValueError(
        f"No normal vehicle composition for hour {hour}"
    )


def build_routes_xml(
    total: int,
    composition: dict[str, float],
) -> tuple[str, dict[str, int]]:
    direction_counts = exact_allocate(
        total,
        DIRECTION_SHARE,
    )

    lines: list[str] = []

    # One route per movement, exactly once.
    for direction in ("N", "S", "E", "W"):
        for turn in ("left", "straight", "right"):
            lines.append(
                f'    <route id="{direction}_{turn}" '
                f'edges="{MOVEMENTS[direction][turn]}" />\n'
            )

    flow_id = 0

    for direction in ("N", "S", "E", "W"):
        movement_counts = exact_allocate(
            direction_counts[direction],
            TURN_SHARE,
        )

        for turn in ("left", "straight", "right"):
            vehicle_counts = exact_allocate(
                movement_counts[turn],
                composition,
            )

            for vtype, number in vehicle_counts.items():
                if number <= 0:
                    continue

                lines.append(
                    f'    <flow id="f_{flow_id}" '
                    f'type="{vtype}" '
                    f'route="{direction}_{turn}" '
                    f'begin="0" end="{DEMAND_DURATION}" '
                    f'number="{number}" '
                    f'departLane="best" '
                    f'departSpeed="max" />\n'
                )
                flow_id += 1

    xml = VTYPES_XML + "".join(lines) + "</routes>\n"
    return xml, direction_counts


def validate_routes(path: Path, expected_total: int) -> None:
    root = ET.parse(path).getroot()

    if root.tag != "routes":
        raise RuntimeError(
            f"{path}: root element is {root.tag!r}, expected 'routes'"
        )

    routes = root.findall("route")
    flows = root.findall("flow")

    route_ids = [r.attrib["id"] for r in routes]
    if len(route_ids) != len(set(route_ids)):
        raise RuntimeError(
            f"{path}: duplicate route IDs detected"
        )

    if len(routes) != 12:
        raise RuntimeError(
            f"{path}: expected 12 routes, found {len(routes)}"
        )

    route_set = set(route_ids)
    total = 0

    for flow in flows:
        route_id = flow.attrib["route"]
        if route_id not in route_set:
            raise RuntimeError(
                f"{path}: flow references missing route {route_id}"
            )
        total += int(flow.attrib["number"])

    if total != expected_total:
        raise RuntimeError(
            f"{path}: expected {expected_total} vehicles, "
            f"found {total}"
        )


def run_sumo(
    cfg: Path,
    tripinfo: Path,
) -> dict[str, float | int]:
    if tripinfo.exists():
        tripinfo.unlink()

    libsumo.start(
        [
            "sumo",
            "-c",
            str(cfg),
            "--seed",
            str(SEED),
            "--no-warnings",
            "true",
        ]
    )

    queue_samples: list[int] = []
    simulation_end = 0.0

    try:
        while libsumo.simulation.getTime() < SIM_END:
            libsumo.simulationStep()
            simulation_end = libsumo.simulation.getTime()

            queue = sum(
                libsumo.edge.getLastStepHaltingNumber(edge)
                for edge in INCOMING_EDGES
            )
            queue_samples.append(queue)
    finally:
        libsumo.close()

    if not tripinfo.exists():
        raise RuntimeError(
            f"SUMO did not create tripinfo: {tripinfo}"
        )

    root = ET.parse(tripinfo).getroot()

    trips = []
    for el in root.findall("tripinfo"):
        arrival = float(el.attrib.get("arrival", "-1"))
        duration = float(el.attrib.get("duration", "-1"))
        trips.append(
            {
                "arrival": arrival,
                "duration": duration,
                "waitingTime": float(
                    el.attrib.get("waitingTime", "nan")
                ),
                "timeLoss": float(
                    el.attrib.get("timeLoss", "nan")
                ),
                "finished": arrival >= 0 and duration >= 0,
            }
        )

    finished = [x for x in trips if x["finished"]]

    travel = [
        x["duration"]
        for x in finished
        if x["duration"] >= 0
    ]
    waiting = [
        x["waitingTime"]
        for x in finished
        if x["waitingTime"] >= 0
    ]
    loss = [
        x["timeLoss"]
        for x in finished
        if x["timeLoss"] >= 0
    ]

    total = len(trips)
    arrived = len(finished)

    return {
        "simulation_end_s": simulation_end,
        "vehicles_reported": total,
        "vehicles_arrived": arrived,
        "vehicles_unfinished": total - arrived,
        "completion_rate_pct": (
            arrived / total * 100
            if total
            else 0.0
        ),
        "mean_travel_time_s": (
            statistics.mean(travel)
            if travel
            else 0.0
        ),
        "mean_waiting_time_s": (
            statistics.mean(waiting)
            if waiting
            else 0.0
        ),
        "mean_time_loss_s": (
            statistics.mean(loss)
            if loss
            else 0.0
        ),
        "mean_queue_vehicles": (
            statistics.mean(queue_samples)
            if queue_samples
            else 0.0
        ),
        "max_queue_vehicles": (
            max(queue_samples)
            if queue_samples
            else 0
        ),
    }


def main() -> None:
    global SEED

    parser = argparse.ArgumentParser(
        description="TrafficPulse vehicle-composition experiment"
    )
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument(
        "--times",
        default=",".join(DEFAULT_TIMES),
        help="Comma-separated HH:MM subset",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="Comma-separated subset of normal,heavy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate files without launching SUMO",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()
    SEED = args.seed

    times = [
        x.strip()
        for x in args.times.split(",")
        if x.strip()
    ]
    scenarios = [
        x.strip()
        for x in args.scenarios.split(",")
        if x.strip()
    ]

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise SystemExit(
                f"Unknown scenario {scenario!r}; "
                f"choose from {SCENARIOS}"
            )

    for required in (
        EVAL_CSV,
        COMPOSITION_FILE,
        NETWORK,
    ):
        if not required.exists():
            raise SystemExit(
                f"Missing required file: {required}"
            )

    DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    predicted_by_time = load_predicted_demands(EVAL_CSV)

    for time_str in times:
        if time_str not in predicted_by_time:
            raise SystemExit(
                f"Time {time_str} not found in {EVAL_CSV}"
            )

    print("=" * 80)
    print("TRAFFICPULSE VEHICLE-COMPOSITION EXPERIMENTS")
    print("=" * 80)
    print(
        f"Date: {args.date} | "
        f"Times: {', '.join(times)}"
    )
    print(
        f"Scenarios: {', '.join(scenarios)}"
    )
    print(
        "Frozen: predicted demand, 1 lane/approach, 50 km/h, "
        "66 s cycle, 20/70/10 turns,"
    )
    print(
        f"        directional shares, seed={SEED}"
    )
    print(f"Dry run: {args.dry_run}")
    print()

    rows: list[dict] = []

    for time_str in times:
        hour = int(time_str.split(":")[0])
        predicted = predicted_by_time[time_str]

        normal = load_normal_composition(hour)

        compositions = {
            "normal": normal,
            "heavy": HEAVY_COMPOSITION,
        }

        for scenario in scenarios:
            composition = compositions[scenario]

            tag = (
                f"comp_{args.date.replace('-', '')}_"
                f"{time_str.replace(':', '')}_"
                f"{scenario}"
            )

            route_file = DEMAND_DIR / f"{tag}.rou.xml"
            config_file = BASE / f"{tag}.sumocfg"
            tripinfo_file = OUTPUT_DIR / f"{tag}.tripinfo.xml"

            route_xml, direction_counts = build_routes_xml(
                predicted,
                composition,
            )
            route_file.write_text(route_xml)

            validate_routes(
                route_file,
                predicted,
            )

            config_file.write_text(
                f"""<configuration>
    <input>
        <net-file value="{NETWORK.resolve()}"/>
        <route-files value="{route_file.resolve()}"/>
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
        <tripinfo-output value="{tripinfo_file.resolve()}"/>
        <tripinfo-output.write-unfinished value="true"/>
    </output>
    <random_number>
        <seed value="{SEED}"/>
    </random_number>
</configuration>
"""
            )

            if args.dry_run:
                print(
                    f"  [dry-run] {time_str} "
                    f"{scenario:<7} "
                    f"demand={predicted} -> "
                    f"{route_file.name}"
                )
                rows.append(
                    {
                        "date": args.date,
                        "time": time_str,
                        "hour": hour,
                        "scenario": scenario,
                        "demand_veh_15min": predicted,
                        "predicted_demand_veh_15min": predicted,
                        "direction_N": direction_counts["N"],
                        "direction_S": direction_counts["S"],
                        "direction_E": direction_counts["E"],
                        "direction_W": direction_counts["W"],
                        "short_share": composition["short"],
                        "towing_share": composition["towing"],
                        "rigid_share": composition["rigid"],
                        "heavy_share": composition["heavy"],
                        "motorcycle_share": composition["motorcycle"],
                        "dry_run": True,
                    }
                )
                continue

            print(
                f"  Running {time_str} "
                f"{scenario:<7} "
                f"demand={predicted} ... ",
                end="",
                flush=True,
            )

            metrics = run_sumo(
                config_file,
                tripinfo_file,
            )

            if metrics["vehicles_reported"] != predicted:
                raise RuntimeError(
                    f"{tag}: expected {predicted} reported vehicles, "
                    f"got {metrics['vehicles_reported']}"
                )

            print(
                f"travel={metrics['mean_travel_time_s']:.1f}s "
                f"wait={metrics['mean_waiting_time_s']:.1f}s "
                f"queue={metrics['mean_queue_vehicles']:.1f}/"
                f"{metrics['max_queue_vehicles']} "
                f"completion={metrics['completion_rate_pct']:.1f}%"
            )

            rows.append(
                {
                    "date": args.date,
                    "time": time_str,
                    "hour": hour,
                    "scenario": scenario,
                    "demand_veh_15min": predicted,
                    "predicted_demand_veh_15min": predicted,
                    "direction_N": direction_counts["N"],
                    "direction_S": direction_counts["S"],
                    "direction_E": direction_counts["E"],
                    "direction_W": direction_counts["W"],
                    "short_share": composition["short"],
                    "towing_share": composition["towing"],
                    "rigid_share": composition["rigid"],
                    "heavy_share": composition["heavy"],
                    "motorcycle_share": composition["motorcycle"],
                    **metrics,
                    "route_file": str(route_file),
                    "config_file": str(config_file),
                    "tripinfo_file": str(tripinfo_file),
                }
            )

    if args.dry_run:
        print()
        print(
            f"Dry run complete: {len(rows)} demand files generated, "
            "no SUMO runs."
        )
        return

    out_csv = RESULT_DIR / "composition_scenarios.csv"
    out_json = RESULT_DIR / "composition_scenarios.json"

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    with out_json.open("w") as f:
        json.dump(
            {
                "date": args.date,
                "times": times,
                "scenarios": scenarios,
                "frozen_config": {
                    "demand_source": (
                        "TrafficPulse predicted representative "
                        "intersection demand"
                    ),
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
                },
                "scenario_definition": {
                    "normal": (
                        "audited TIRTL hourly vehicle composition"
                    ),
                    "heavy": (
                        "short 75%, towing 2%, rigid 15%, "
                        "heavy articulated 7%, motorcycle 1%"
                    ),
                },
                "runs": rows,
            },
            f,
            indent=2,
        )

    if len(rows) != len(times) * len(scenarios):
        raise RuntimeError(
            f"Expected {len(times) * len(scenarios)} runs, "
            f"got {len(rows)}"
        )

    if any(
        float(r["completion_rate_pct"]) != 100.0
        or int(r["vehicles_unfinished"]) != 0
        for r in rows
    ):
        raise RuntimeError(
            "At least one composition run did not complete "
            "100% of vehicles."
        )

    print()
    print("=" * 80)
    print(
        f"COMPOSITION EXPERIMENTS COMPLETE: {len(rows)} runs"
    )
    print(f"CSV : {out_csv}")
    print(f"JSON: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
