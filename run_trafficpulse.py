from pathlib import Path
import json
import statistics
import xml.etree.ElementTree as ET

import libsumo


# ============================================================
# Paths
# ============================================================

BASE = Path("sumo/trafficpulse").resolve()

CFG = BASE / "trafficpulse_baseline.sumocfg"

TRIPINFO = BASE / "output" / "tripinfo.xml"

RESULT_DIR = BASE / "output"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Simulation constants
# ============================================================

SIM_END = 3600

INCOMING_EDGES = [
    "N_C",
    "S_C",
    "E_C",
    "W_C",
]


# ============================================================
# Helpers
# ============================================================

def read_tripinfo(path: Path) -> list[dict]:
    """
    Read SUMO tripinfo output.

    Both completed and unfinished trips are included because
    the config enables tripinfo-output.write-unfinished.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"TripInfo output was not created: {path}"
        )

    root = ET.parse(path).getroot()

    trips = []

    for element in root.findall("tripinfo"):

        trip = {
            "id": element.attrib.get("id"),
            "depart": float(
                element.attrib.get("depart", "nan")
            ),
            "arrival": float(
                element.attrib.get("arrival", "nan")
            ),
            "duration": float(
                element.attrib.get("duration", "nan")
            ),
            "waitingTime": float(
                element.attrib.get("waitingTime", "nan")
            ),
            "timeLoss": float(
                element.attrib.get("timeLoss", "nan")
            ),
            "routeLength": float(
                element.attrib.get("routeLength", "nan")
            ),
            "vtype": element.attrib.get(
                "vtype",
                "",
            ),
        }

        # SUMO uses negative arrival/duration values when
        # an unfinished trip is written.
        trip["finished"] = (
            trip["arrival"] >= 0
            and trip["duration"] >= 0
        )

        trips.append(trip)

    return trips


# ============================================================
# Main
# ============================================================

def main():

    if not CFG.exists():
        raise FileNotFoundError(
            f"SUMO configuration does not exist: {CFG}"
        )

    print("=" * 80)
    print("TRAFFICPULSE SUMO BASELINE")
    print("=" * 80)

    print(f"Configuration: {CFG}")
    print("Starting libsumo...")

    # Remove stale TripInfo output.
    if TRIPINFO.exists():
        TRIPINFO.unlink()

    libsumo.start(
        [
            "sumo",
            "-c",
            str(CFG),
            "--seed",
            "42",
        ]
    )

    queue_samples = []

    simulation_end = 0.0

    try:

        # Run through the complete configured horizon.
        while libsumo.simulation.getTime() < SIM_END:

            libsumo.simulationStep()

            simulation_end = (
                libsumo.simulation.getTime()
            )

            # Number of halted vehicles on incoming edges.
            queue = sum(
                libsumo.edge.getLastStepHaltingNumber(
                    edge
                )
                for edge in INCOMING_EDGES
            )

            queue_samples.append(queue)

    finally:

        libsumo.close()

    # ========================================================
    # TripInfo
    # ========================================================

    trips = read_tripinfo(TRIPINFO)

    finished = [
        trip
        for trip in trips
        if trip["finished"]
    ]

    unfinished = [
        trip
        for trip in trips
        if not trip["finished"]
    ]

    # ========================================================
    # Metrics
    # ========================================================

    finished_travel_times = [
        t["duration"]
        for t in finished
        if t["duration"] >= 0
    ]

    finished_waiting = [
        t["waitingTime"]
        for t in finished
        if t["waitingTime"] >= 0
    ]

    finished_time_loss = [
        t["timeLoss"]
        for t in finished
        if t["timeLoss"] >= 0
    ]

    all_valid_waiting = [
        t["waitingTime"]
        for t in trips
        if t["waitingTime"] >= 0
    ]

    mean_travel_time = (
        statistics.mean(
            finished_travel_times
        )
        if finished_travel_times
        else 0.0
    )

    mean_waiting_time = (
        statistics.mean(
            finished_waiting
        )
        if finished_waiting
        else 0.0
    )

    mean_time_loss = (
        statistics.mean(
            finished_time_loss
        )
        if finished_time_loss
        else 0.0
    )

    mean_queue = (
        statistics.mean(queue_samples)
        if queue_samples
        else 0.0
    )

    max_queue = (
        max(queue_samples)
        if queue_samples
        else 0
    )

    total_vehicles = len(trips)

    throughput = (
        len(finished)
        / total_vehicles
        * 100.0
        if total_vehicles
        else 0.0
    )

    # ========================================================
    # Results
    # ========================================================

    results = {
        "scenario": "baseline",

        "simulation_end_s": simulation_end,

        "vehicles_reported": total_vehicles,

        "vehicles_arrived": len(finished),

        "vehicles_unfinished": len(unfinished),

        "completion_rate_pct": throughput,

        "mean_travel_time_s": mean_travel_time,

        "mean_waiting_time_s": mean_waiting_time,

        "mean_time_loss_s": mean_time_loss,

        "mean_queue_vehicles": mean_queue,

        "max_queue_vehicles": max_queue,
    }

    output = RESULT_DIR / "baseline_results.json"

    output.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    # Also save vehicle-level trip data for later analysis.
    trip_csv = RESULT_DIR / "baseline_tripinfo.csv"

    with trip_csv.open("w") as f:

        f.write(
            "id,depart,arrival,duration,"
            "waitingTime,timeLoss,routeLength,"
            "vtype,finished\n"
        )

        for trip in trips:

            f.write(
                f"{trip['id']},"
                f"{trip['depart']},"
                f"{trip['arrival']},"
                f"{trip['duration']},"
                f"{trip['waitingTime']},"
                f"{trip['timeLoss']},"
                f"{trip['routeLength']},"
                f"{trip['vtype']},"
                f"{trip['finished']}\n"
            )

    print()
    print("=" * 80)
    print("TRAFFICPULSE BASELINE RESULTS")
    print("=" * 80)

    for key, value in results.items():
        print(
            f"{key:<30} {value}"
        )

    print()
    print(f"TripInfo: {TRIPINFO}")
    print(f"Trip CSV: {trip_csv}")
    print(f"Results:  {output}")


if __name__ == "__main__":
    main()
