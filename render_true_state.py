#!/usr/bin/env python3
"""Render TRUE-state intersection images from captured SUMO vehicle states.

Every dot = one real simulated vehicle at its real (x, y) at the stated
timestep, colored by the simulation's own vType color. Counts and halting
numbers are printed on each image and match states.json exactly.

Styling (dark background, round markers) is presentation; positions,
counts, classes, scenario, timestep are data. Marker size is fixed and
NOT to scale; road width is exaggerated for visibility (as in SUMO-GUI).

Run with: python3 render_true_state.py   (needs matplotlib)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parent
IN_DIR = PROJECT / "artifacts/figures/trafficpulse/true_state"

BG = "#07131F"
PANEL = "#0A2136"
PANEL_EDGE = "#1D4E6E"
ROAD = "#526A7B"
ROAD_EDGE = "#354A5B"
ROAD_JUNCTION = "#607789"
LABEL = "#E8F4FA"
DIM = "#7FA3B8"

HALF = 260.0       # viewport half-size (m); network arms are +/-250 m
ROAD_W = 10.0      # drawn road width (m) — exaggerated, see method note
JR = 9.0           # drawn junction half-size (m)


def junction_center() -> tuple[float, float]:
    """True junction position read from the built network (netconvert applies
    an offset, so sim coordinates are NOT the .nod.xml coordinates)."""
    root = ET.parse(
        PROJECT / "sumo/trafficpulse/network/intersection.net.xml").getroot()
    for j in root.iter("junction"):
        if j.attrib.get("id") == "C":
            return float(j.attrib["x"]), float(j.attrib["y"])
    raise RuntimeError("Junction C not found in network")


CX, CY = junction_center()
print(f"junction C at sim ({CX}, {CY}) — centering viewport on it")


def draw(ax, state: dict) -> None:
    ax.set_facecolor(BG)
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF * 0.625, HALF * 0.625)
    ax.set_aspect("equal")
    ax.axis("off")
    # road: two straight bars, true orientation and true extent
    for (x, y, w, h) in [(-ROAD_W / 2, -250, ROAD_W, 500),
                         (-250, -ROAD_W / 2, 500, ROAD_W)]:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2,rounding_size=2",
            facecolor=ROAD_EDGE, edgecolor="none", zorder=2))
    for (x, y, w, h) in [(-ROAD_W / 2 + 1, -250, ROAD_W - 2, 500),
                         (-250, -ROAD_W / 2 + 1, 500, ROAD_W - 2)]:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=1.5",
            facecolor=ROAD, edgecolor="none", zorder=3))
    ax.add_patch(FancyBboxPatch(
        (-JR, -JR), 2 * JR, 2 * JR,
        boxstyle="round,pad=0.1,rounding_size=1.2",
        facecolor=ROAD_JUNCTION, edgecolor="none", alpha=0.9, zorder=4))
    # vehicles: TRUE positions recentered on the junction, sim's own colors
    colors = state["vtype_colors"]
    for v in state["vehicles"]:
        c = colors[v["type"]]
        ax.add_patch(Circle((v["x"] - CX, v["y"] - CY), 5.5, color=c, alpha=0.25,
                            linewidth=0, zorder=5))
    for v in state["vehicles"]:
        c = colors[v["type"]]
        ax.add_patch(Circle((v["x"] - CX, v["y"] - CY), 3.4, color=c, alpha=0.98,
                            ec="white", lw=0.7, zorder=6))
    for txt, pos in [("N", (16, 238)), ("S", (16, -238)),
                     ("E", (238, 14)), ("W", (-238, 14))]:
        ax.text(pos[0], pos[1], txt, color=LABEL, fontsize=9,
                fontweight="bold", ha="center", va="center", zorder=7)


def main() -> None:
    states = json.loads((IN_DIR / "states.json").read_text())
    manifest: dict = {
        "method": (
            "Each dot marks one real SUMO vehicle at its simulated (x, y) "
            "at the stated timestep (seed 42). Colors are the simulation's "
            "own vType colors. Counts and halting numbers are read from the "
            "simulation and printed on each image. Marker size is fixed "
            "(not to scale); road width is exaggerated for visibility."
        ),
        "images": {},
    }
    for key, st in states.items():
        n = len(st["vehicles"])
        assert n == st["on_network_count"], f"{key}: count mismatch"
        fig, ax = plt.subplots(figsize=(12, 7.5), dpi=200)
        fig.patch.set_facecolor(BG)
        draw(ax, st)
        fig.text(0.5, 0.93, st["label"], color=LABEL, fontsize=18,
                 fontweight="bold", ha="center", va="center")
        fig.text(0.5, 0.885,
                 f"t = {st['timestep_s']:.0f} s  ·  "
                 f"{n} vehicles on network  ·  "
                 f"{st['halting_queue_at_t']} halting on approaches",
                 color=LABEL, fontsize=12, ha="center", va="center")
        fig.text(0.5, 0.06,
                 f"True SUMO state · {st['sumocfg']} · seed {st['seed']} · "
                 "markers not to scale",
                 color=DIM, fontsize=8, ha="center", va="center")
        p = IN_DIR / f"true_{key}.png"
        fig.savefig(p, facecolor=BG)
        plt.close(fig)
        tc: dict[str, int] = {}
        for v in st["vehicles"]:
            tc[v["type"]] = tc.get(v["type"], 0) + 1
        manifest["images"][key] = {
            "file": f"true_{key}.png",
            "label": st["label"],
            "sumocfg": st["sumocfg"],
            "demand_veh_15min": st["demand_veh_15min"],
            "timestep_s": st["timestep_s"],
            "on_network_count": n,
            "halting_queue_at_t": st["halting_queue_at_t"],
            "type_counts": tc,
            "vtype_colors": st["vtype_colors"],
        }
        print(f"wrote {p.name}  on-network={n:4d} halting={st['halting_queue_at_t']:3d}")
    (IN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote manifest.json ({len(manifest['images'])} images)")


if __name__ == "__main__":
    main()
