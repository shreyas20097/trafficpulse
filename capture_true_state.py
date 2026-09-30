#!/usr/bin/env python3
"""Capture TRUE vehicle states from real SUMO runs for the deck visuals.

Runs the actual 17:15 experiment scenarios headless (seed 42, same as the
experiments behind the dashboard numbers) and records every vehicle's
real (x, y, type) position at a stated timestep via libsumo.

No invented positions, no invented counts, no invented classes.
Output: artifacts/figures/trafficpulse/true_state/states.json

Run with: .sumo-venv/bin/python capture_true_state.py
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import libsumo

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "artifacts/figures/trafficpulse/true_state"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

VOL = PROJECT / "sumo/trafficpulse/volume"
COMP = PROJECT / "sumo/trafficpulse/composition"
CAP = PROJECT / "sumo/trafficpulse/capacity"

# (key, sumocfg, demand veh/15min, mode, label)
# mode = ("early", T) -> state at fixed early timestep T (formation phase)
# mode = ("fixed", T) -> state at fixed timestep T (cross-scenario comparable)
SCENARIOS = [
    ("prototype_sparse", VOL / "vol_20260117_1715_low.sumocfg",
     802, ("early", 60), "17:15 low demand (802 veh/15 min)"),
    ("inflow", VOL / "vol_20260117_1715_predicted.sumocfg",
     1002, ("early", 90), "17:15 predicted demand (1002 veh/15 min)"),
    ("demand_low", VOL / "vol_20260117_1715_low.sumocfg",
     802, ("fixed", 1500), "17:15 low demand (802 veh/15 min)"),
    ("demand_base", VOL / "vol_20260117_1715_predicted.sumocfg",
     1002, ("fixed", 1500), "17:15 predicted demand (1002 veh/15 min)"),
    ("demand_high", VOL / "vol_20260117_1715_high.sumocfg",
     1202, ("fixed", 1500), "17:15 high demand (1202 veh/15 min)"),
    ("demand_peak", VOL / "vol_20260117_1715_very_high.sumocfg",
     1403, ("fixed", 1500), "17:15 very-high demand (1403 veh/15 min)"),
    ("composition_normal", COMP / "comp_20260117_1715_normal.sumocfg",
     1002, ("fixed", 1500), "17:15 normal TIRTL mix (1002 veh/15 min)"),
    ("composition_heavy", COMP / "comp_20260117_1715_heavy.sumocfg",
     1002, ("fixed", 1500), "17:15 heavy mix (1002 veh/15 min)"),
    ("capacity_1lane", CAP / "cap_20260117_1715_1lane.sumocfg",
     1002, ("fixed", 1500), "17:15 predicted demand, 1 lane/approach (1002 veh/15 min)"),
    ("capacity_2lane", CAP / "cap_20260117_1715_2lane.sumocfg",
     1002, ("fixed", 1500), "17:15 predicted demand, 2 lanes/approach (1002 veh/15 min)"),
    ("conclusion", VOL / "vol_20260117_1715_predicted.sumocfg",
     1002, ("early", 150), "17:15 predicted demand (1002 veh/15 min)"),
]

INCOMING_EDGES = ["N_C", "S_C", "E_C", "W_C"]
QUEUE_WINDOW_S = 900


def vtype_colors(cfg: Path) -> dict[str, str]:
    """Read the sim's own vType color attributes from its route file."""
    root = ET.parse(str(cfg)).getroot()
    ns = {"s": "http://www.w3.org/2001/XMLSchema-instance"}
    colors: dict[str, str] = {}
    for inp in root.iter("route-files"):
        pass
    # route-files value lives in <route-files value="..."/>
    route_file = None
    for el in root.iter():
        if el.tag == "route-files" and "value" in el.attrib:
            route_file = el.attrib["value"]
    if route_file is None:
        raise RuntimeError(f"No route-files in {cfg}")
    rroot = ET.parse(route_file).getroot()
    for vt in rroot.findall("vType"):
        vid = vt.attrib["id"]
        col = vt.attrib.get("color", "255,255,0")
        parts = [int(float(c)) for c in col.split(",")[:3]]
        colors[vid] = "#%02X%02X%02X" % tuple(parts)
    return colors


def capture(cfg: Path, timestep: float) -> tuple[list[dict], int]:
    """Run to `timestep`, return (vehicles, halting queue on incoming edges)."""
    libsumo.start(["sumo", "-c", str(cfg), "--seed", str(SEED),
                   "--no-warnings", "true"])
    try:
        while libsumo.simulation.getTime() < timestep:
            libsumo.simulationStep()
        out = []
        for vid in libsumo.vehicle.getIDList():
            x, y = libsumo.vehicle.getPosition(vid)
            out.append({"id": vid, "x": float(x), "y": float(y),
                        "type": libsumo.vehicle.getTypeID(vid)})
        queue = sum(libsumo.edge.getLastStepHaltingNumber(e)
                    for e in INCOMING_EDGES)
        return out, int(queue)
    finally:
        libsumo.close()


def max_queue_timestep(cfg: Path) -> tuple[float, int]:
    """First pass: halting queue per step over the demand window; return
    (argmax timestep, max queue). Deterministic under fixed seed."""
    libsumo.start(["sumo", "-c", str(cfg), "--seed", str(SEED),
                   "--no-warnings", "true"])
    try:
        best_t, best_q = 0.0, -1
        while libsumo.simulation.getTime() < QUEUE_WINDOW_S:
            libsumo.simulationStep()
            q = sum(libsumo.edge.getLastStepHaltingNumber(e)
                    for e in INCOMING_EDGES)
            if int(q) > best_q:
                best_q, best_t = int(q), float(libsumo.simulation.getTime())
        return best_t, best_q
    finally:
        libsumo.close()


def main() -> None:
    states: dict = {}
    for key, cfg, demand, mode, label in SCENARIOS:
        if not cfg.exists():
            raise FileNotFoundError(f"Missing scenario config: {cfg}")
        t = float(mode[1])
        selection = f"t={t:.0f}s ({'formation' if mode[0] == 'early' else 'drain, cross-scenario comparable'})"
        vehicles, queue = capture(cfg, t)
        colors = vtype_colors(cfg)
        unknown = {v["type"] for v in vehicles} - set(colors)
        if unknown:
            raise RuntimeError(f"{key}: vehicle types without sim colors: {unknown}")
        states[key] = {
            "label": label,
            "sumocfg": str(cfg.relative_to(PROJECT)),
            "demand_veh_15min": demand,
            "timestep_s": t,
            "state_selection": selection,
            "halting_queue_at_t": queue,
            "seed": SEED,
            "on_network_count": len(vehicles),
            "vtype_colors": colors,
            "vehicles": vehicles,
        }
        type_counts: dict[str, int] = {}
        for v in vehicles:
            type_counts[v["type"]] = type_counts.get(v["type"], 0) + 1
        print(f"{key:20s} {selection:28s} on-network={len(vehicles):4d} "
              f"halting={queue:3d} types={type_counts}")
    out = OUT_DIR / "states.json"
    out.write_text(json.dumps(states, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
