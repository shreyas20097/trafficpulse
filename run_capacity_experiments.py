#!/usr/bin/env python3
"""
TrafficPulse capacity-sensitivity experiment.

Independent variable:
    Number of lanes per approach: 1 vs 2.

Frozen:
    TrafficPulse predicted demand at each evaluation time
    Vehicle composition = audited TIRTL hourly composition
    50 km/h
    66 s signal cycle
    20/70/10 turn shares
    directional shares
    SUMO seed

Runs:
    8 times x 2 capacities = 16 SUMO runs.

The original network XML is never modified. A derived two-lane network
is written under sumo/trafficpulse/capacity/network/.

Outputs:
    artifacts/experiments/capacity/capacity_scenarios.csv
    artifacts/experiments/capacity/capacity_scenarios.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import libsumo


PROJECT = Path(__file__).resolve().parent

EVAL_CSV = PROJECT / "artifacts/evaluation/trafficpulse_evaluation.csv"
COMPOSITION_FILE = PROJECT / "artifacts/tirtl_vehicle_composition_by_hour.csv"

ORIGINAL_NETWORK = (
    PROJECT
    / "sumo/trafficpulse/network/intersection.net.xml"
)

# Plain-XML edge/node sources used to (re)build networks. The compiled
# .net.xml stores lanes as child <lane> elements, so lane count can only
# be changed by rebuilding via netconvert from these sources.
NETWORK_NODES = (
    PROJECT
    / "sumo/trafficpulse/network/intersection.nod.xml"
)

NETWORK_EDGES = (
    PROJECT
    / "sumo/trafficpulse/network/intersection.edg.xml"
)

NETCONVERT = PROJECT / ".sumo-venv/bin/netconvert"

# netconvert flags frozen to the original baseline build
# (see build_trafficpulse_sumo.py): static TLS, opposites layout,
# 66 s cycle, 3 s yellow, no protected-left phase.
NETCONVERT_TLS_FLAGS = [
    "--tls.default-type", "static",
    "--tls.layout", "opposites",
    "--tls.cycle.time", "66",
    "--tls.yellow.time", "3",
    "--tls.left-green.time", "0",
]

BASE = PROJECT / "sumo/trafficpulse/capacity"
NETWORK_DIR = BASE / "network"
DEMAND_DIR = BASE / "demand"
OUTPUT_DIR = BASE / "output"

RESULT_DIR = (
    PROJECT
    / "artifacts/experiments/capacity"
)

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

INCOMING_EDGES = [
    "N_C",
    "S_C",
    "E_C",
    "W_C",
]

# These are the physical approach/departure edges in the simple
# TrafficPulse network. Internal junction edges are not modified.
CAPACITY_EDGES = [
    "N_C",
    "S_C",
    "E_C",
    "W_C",
    "C_N",
    "C_S",
    "C_E",
    "C_W",
]

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

CAPACITIES = (1, 2)


def exact_allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    keys = list(shares)
    total_share = sum(shares.values())

    if not math.isclose(
        total_share,
        1.0,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"Shares must sum to 1.0; got {total_share}"
        )

    raw = {
        key: total * shares[key]
        for key in keys
    }

    values = {
        key: math.floor(raw[key])
        for key in keys
    }

    remaining = total - sum(values.values())

    order = sorted(
        keys,
        key=lambda key: (
            raw[key] - values[key]
        ),
        reverse=True,
    )

    for key in order[:remaining]:
        values[key] += 1

    assert sum(values.values()) == total
    return values


def load_predicted_demands(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(path)

    result: dict[str, int] = {}

    with path.open() as f:
        for row in csv.DictReader(f):
            result[row["time"].strip()] = int(
                round(
                    float(
                        row[
                            "predicted_representative_demand_veh_15min"
                        ]
                    )
                )
            )

    return result


def load_composition(hour: int) -> dict[str, float]:
    if not COMPOSITION_FILE.exists():
        raise FileNotFoundError(COMPOSITION_FILE)

    with COMPOSITION_FILE.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw_hour = row.get("hour", "")

            if not raw_hour:
                raw_hour = list(row.values())[0]

            if int(float(raw_hour)) != int(hour):
                continue

            comp = {}

            for source, sumo_type in COMPOSITION_MAP.items():
                comp[sumo_type] = float(
                    row[f"{source}_share"]
                )

            total = sum(comp.values())

            if total <= 0:
                raise RuntimeError(
                    f"Invalid composition for hour {hour}"
                )

            return {
                key: value / total
                for key, value in comp.items()
            }

    raise ValueError(
        f"No vehicle composition for hour {hour}"
    )


def make_two_lane_network(
    source: Path,
    destination: Path,
) -> None:
    """
    Create a derived 2-lane network without modifying the original.

    The compiled .net.xml cannot be lane-edited in place (lanes are
    child elements with geometry), so the network is rebuilt with
    netconvert from the original .nod.xml/.edg.xml sources, with only
    numLanes changed 1 -> 2 on the eight physical approach/departure
    edges. TLS settings are frozen to the baseline build flags.
    The `source` argument is the original .net.xml (used only to
    locate/verify against the baseline); the build uses the XML
    edge/node sources.
    """

    for required in (NETWORK_NODES, NETWORK_EDGES, NETCONVERT):
        if not required.exists():
            raise FileNotFoundError(required)

    edge_tree = ET.parse(NETWORK_EDGES)
    edge_root = edge_tree.getroot()

    changed = []

    for edge in edge_root.findall("edge"):
        edge_id = edge.attrib.get("id")

        if edge_id in CAPACITY_EDGES:
            old = edge.attrib.get("numLanes")

            if old is None:
                raise RuntimeError(
                    f"Edge {edge_id} has no numLanes attribute"
                )

            edge.set("numLanes", "2")
            changed.append(edge_id)

    if set(changed) != set(CAPACITY_EDGES):
        missing = sorted(set(CAPACITY_EDGES) - set(changed))
        raise RuntimeError(
            "Could not convert all capacity edges to two lanes. "
            f"Missing: {missing}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    two_lane_edg = destination.with_suffix(".edg.xml")
    edge_tree.write(
        two_lane_edg,
        encoding="utf-8",
        xml_declaration=True,
    )

    cmd = [
        str(NETCONVERT),
        "--node-files", str(NETWORK_NODES),
        "--edge-files", str(two_lane_edg),
        "--output-file", str(destination),
        *NETCONVERT_TLS_FLAGS,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"netconvert failed:\n{result.stdout}"
        )

    # Verify lane counts on the rebuilt network.
    verify = ET.parse(destination).getroot()

    for edge_id in CAPACITY_EDGES:
        matches = [
            e
            for e in verify.findall("edge")
            if e.attrib.get("id") == edge_id
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Verification failed for edge {edge_id}"
            )

        lanes = matches[0].findall("lane")

        if len(lanes) != 2:
            raise RuntimeError(
                f"Verification failed: {edge_id} has "
                f"{len(lanes)} lanes, expected 2"
            )

    print(
        f"Created 2-lane network: {destination}"
    )


def build_routes_xml(
    total: int,
    composition: dict[str, float],
) -> tuple[str, dict[str, int]]:
    direction_counts = exact_allocate(
        total,
        DIRECTION_SHARE,
    )

    lines = []

    for direction in (
        "N",
        "S",
        "E",
        "W",
    ):
        for turn in (
            "left",
            "straight",
            "right",
        ):
            lines.append(
                f'    <route '
                f'id="{direction}_{turn}" '
                f'edges="{MOVEMENTS[direction][turn]}" />\n'
            )

    flow_id = 0

    for direction in (
        "N",
        "S",
        "E",
        "W",
    ):
        movement_counts = exact_allocate(
            direction_counts[direction],
            TURN_SHARE,
        )

        for turn in (
            "left",
            "straight",
            "right",
        ):
            vehicle_counts = exact_allocate(
                movement_counts[turn],
                composition,
            )

            for vehicle_type, number in (
                vehicle_counts.items()
            ):
                if number <= 0:
                    continue

                lines.append(
                    f'    <flow '
                    f'id="f_{flow_id}" '
                    f'type="{vehicle_type}" '
                    f'route="{direction}_{turn}" '
                    f'begin="0" '
                    f'end="{DEMAND_DURATION}" '
                    f'number="{number}" '
                    f'departLane="best" '
                    f'departSpeed="max" />\n'
                )

                flow_id += 1

    return (
        VTYPES_XML
        + "".join(lines)
        + "</routes>\n",
        direction_counts,
    )


def validate_routes(
    route_file: Path,
    expected_total: int,
) -> None:
    root = ET.parse(route_file).getroot()

    if root.tag != "routes":
        raise RuntimeError(
            f"Invalid root element in {route_file}"
        )

    routes = root.findall("route")
    flows = root.findall("flow")

    if len(routes) != 12:
        raise RuntimeError(
            f"Expected 12 routes; found {len(routes)}"
        )

    route_ids = {
        route.attrib["id"]
        for route in routes
    }

    if len(route_ids) != 12:
        raise RuntimeError(
            "Duplicate route IDs detected"
        )

    total = 0

    for flow in flows:
        if flow.attrib["route"] not in route_ids:
            raise RuntimeError(
                "Flow references an unknown route"
            )

        total += int(
            flow.attrib["number"]
        )

    if total != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} vehicles; "
            f"found {total}"
        )


def read_tripinfo(
    path: Path,
) -> list[dict]:
    if not path.exists():
        raise RuntimeError(
            f"Missing tripinfo file: {path}"
        )

    root = ET.parse(path).getroot()

    trips = []

    for element in root.findall("tripinfo"):
        arrival = float(
            element.attrib.get(
                "arrival",
                "-1",
            )
        )

        duration = float(
            element.attrib.get(
                "duration",
                "-1",
            )
        )

        trips.append(
            {
                "arrival": arrival,
                "duration": duration,
                "waitingTime": float(
                    element.attrib.get(
                        "waitingTime",
                        "nan",
                    )
                ),
                "timeLoss": float(
                    element.attrib.get(
                        "timeLoss",
                        "nan",
                    )
                ),
                "finished": (
                    arrival >= 0
                    and duration >= 0
                ),
            }
        )

    return trips


def run_sumo(
    config_file: Path,
    tripinfo_file: Path,
) -> dict:
    if tripinfo_file.exists():
        tripinfo_file.unlink()

    libsumo.start(
        [
            "sumo",
            "-c",
            str(config_file),
            "--seed",
            str(SEED),
            "--no-warnings",
            "true",
        ]
    )

    queue_samples = []
    sim_end = 0.0

    try:
        while (
            libsumo.simulation.getTime()
            < SIM_END
        ):
            libsumo.simulationStep()

            sim_end = (
                libsumo.simulation.getTime()
            )

            queue = sum(
                libsumo.edge.getLastStepHaltingNumber(
                    edge_id
                )
                for edge_id in INCOMING_EDGES
            )

            queue_samples.append(queue)

    finally:
        libsumo.close()

    trips = read_tripinfo(
        tripinfo_file
    )

    finished = [
        trip
        for trip in trips
        if trip["finished"]
    ]

    travel = [
        trip["duration"]
        for trip in finished
        if trip["duration"] >= 0
    ]

    waiting = [
        trip["waitingTime"]
        for trip in finished
        if trip["waitingTime"] >= 0
    ]

    time_loss = [
        trip["timeLoss"]
        for trip in finished
        if trip["timeLoss"] >= 0
    ]

    total = len(trips)
    arrived = len(finished)

    return {
        "simulation_end_s": sim_end,
        "vehicles_reported": total,
        "vehicles_arrived": arrived,
        "vehicles_unfinished": (
            total - arrived
        ),
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
            statistics.mean(time_loss)
            if time_loss
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
        description=(
            "TrafficPulse capacity "
            "experiment"
        )
    )

    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
    )

    parser.add_argument(
        "--times",
        default=",".join(
            DEFAULT_TIMES
        ),
        help=(
            "Comma-separated "
            "HH:MM subset"
        ),
    )

    parser.add_argument(
        "--capacities",
        default="1,2",
        help=(
            "Comma-separated "
            "lane counts"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate network/demand "
            "files without running SUMO"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()
    SEED = args.seed

    try:
        capacities = [
            int(x.strip())
            for x in args.capacities.split(",")
            if x.strip()
        ]
    except ValueError as exc:
        raise SystemExit(
            "Invalid --capacities."
        ) from exc

    if not capacities:
        raise SystemExit(
            "At least one capacity is required."
        )

    for capacity in capacities:
        if capacity not in (1, 2):
            raise SystemExit(
                "Only 1 and 2 lane "
                "capacities are supported."
            )

    times = [
        x.strip()
        for x in args.times.split(",")
        if x.strip()
    ]

    for path in (
        EVAL_CSV,
        COMPOSITION_FILE,
        ORIGINAL_NETWORK,
    ):
        if not path.exists():
            raise SystemExit(
                f"Missing required file: {path}"
            )

    NETWORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    DEMAND_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predicted_by_time = (
        load_predicted_demands(EVAL_CSV)
    )

    for time_str in times:
        if time_str not in predicted_by_time:
            raise SystemExit(
                f"Time {time_str} is not present "
                f"in {EVAL_CSV}"
            )

    one_lane_network = (
        ORIGINAL_NETWORK
    )

    two_lane_network = (
        NETWORK_DIR
        / "intersection_2lane.net.xml"
    )

    if 2 in capacities:
        if (
            not two_lane_network.exists()
            or two_lane_network.stat().st_mtime
            < ORIGINAL_NETWORK.stat().st_mtime
        ):
            make_two_lane_network(
                ORIGINAL_NETWORK,
                two_lane_network,
            )
        else:
            print(
                f"Using existing verified 2-lane "
                f"network: {two_lane_network}"
            )

    print("=" * 80)
    print("TRAFFICPULSE CAPACITY-SENSITIVITY EXPERIMENTS")
    print("=" * 80)
    print(
        f"Date: {args.date} | "
        f"Times: {', '.join(times)}"
    )
    print(
        f"Capacities: "
        f"{', '.join(map(str, capacities))} lanes/approach"
    )
    print(
        "Frozen: TrafficPulse demand, TIRTL composition, "
        "50 km/h, 66 s cycle, 20/70/10 turns,"
    )
    print(
        f"        directional shares, seed={SEED}"
    )
    print(f"Dry run: {args.dry_run}")
    print()

    rows: list[dict] = []

    for time_str in times:
        hour = int(
            time_str.split(":")[0]
        )

        demand = predicted_by_time[
            time_str
        ]

        composition = load_composition(
            hour
        )

        for lanes in capacities:
            network = (
                one_lane_network
                if lanes == 1
                else two_lane_network
            )

            tag = (
                f"cap_{args.date.replace('-', '')}_"
                f"{time_str.replace(':', '')}_"
                f"{lanes}lane"
            )

            route_file = (
                DEMAND_DIR
                / f"{tag}.rou.xml"
            )

            config_file = (
                BASE
                / f"{tag}.sumocfg"
            )

            tripinfo_file = (
                OUTPUT_DIR
                / f"{tag}.tripinfo.xml"
            )

            route_xml, direction_counts = (
                build_routes_xml(
                    demand,
                    composition,
                )
            )

            route_file.write_text(
                route_xml
            )

            validate_routes(
                route_file,
                demand,
            )

            config_file.write_text(
                f"""<configuration>
    <input>
        <net-file value="{network.resolve()}"/>
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
                    f"{lanes} lane "
                    f"demand={demand}"
                )

                rows.append(
                    {
                        "date": args.date,
                        "time": time_str,
                        "hour": hour,
                        "lanes_per_approach": lanes,
                        "demand_veh_15min": demand,
                        "predicted_demand_veh_15min": demand,
                        "direction_N": direction_counts["N"],
                        "direction_S": direction_counts["S"],
                        "direction_E": direction_counts["E"],
                        "direction_W": direction_counts["W"],
                        "dry_run": True,
                    }
                )

                continue

            print(
                f"  Running {time_str} "
                f"{lanes} lane "
                f"demand={demand} ... ",
                end="",
                flush=True,
            )

            metrics = run_sumo(
                config_file,
                tripinfo_file,
            )

            if (
                metrics["vehicles_reported"]
                != demand
            ):
                raise RuntimeError(
                    f"{tag}: SUMO reported "
                    f"{metrics['vehicles_reported']} "
                    f"vehicles, expected {demand}"
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
                    "lanes_per_approach": lanes,
                    "demand_veh_15min": demand,
                    "predicted_demand_veh_15min": demand,
                    "direction_N": direction_counts["N"],
                    "direction_S": direction_counts["S"],
                    "direction_E": direction_counts["E"],
                    "direction_W": direction_counts["W"],
                    **metrics,
                    "network_file": str(network),
                    "route_file": str(route_file),
                    "config_file": str(config_file),
                    "tripinfo_file": str(tripinfo_file),
                }
            )

    if args.dry_run:
        print()
        print(
            f"Dry run complete: {len(rows)} "
            "scenario files generated; "
            "no SUMO simulations run."
        )
        return

    expected_rows = (
        len(times) * len(capacities)
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"got {len(rows)}"
        )

    for row in rows:
        if (
            row["vehicles_reported"]
            != row["demand_veh_15min"]
        ):
            raise RuntimeError(
                "Vehicle-count mismatch detected."
            )

        if (
            row["vehicles_unfinished"] != 0
            or row["completion_rate_pct"] != 100.0
        ):
            raise RuntimeError(
                "At least one capacity run did not "
                "complete 100% of vehicles."
            )

    out_csv = (
        RESULT_DIR
        / "capacity_scenarios.csv"
    )

    out_json = (
        RESULT_DIR
        / "capacity_scenarios.json"
    )

    with out_csv.open(
        "w",
        newline="",
    ) as f:
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
                "capacities": capacities,
                "frozen_config": {
                    "demand_source": (
                        "TrafficPulse predicted "
                        "representative intersection demand"
                    ),
                    "vehicle_composition_source": (
                        "audited TIRTL hourly composition"
                    ),
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
                "capacity_definition": {
                    "one_lane": (
                        "original TrafficPulse network"
                    ),
                    "two_lane": (
                        "derived network with N_C, S_C, "
                        "E_C, W_C, C_N, C_S, C_E, C_W "
                        "set to numLanes=2"
                    ),
                },
                "runs": rows,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        f"CAPACITY EXPERIMENTS COMPLETE: "
        f"{len(rows)} runs"
    )
    print(f"CSV : {out_csv}")
    print(f"JSON: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
