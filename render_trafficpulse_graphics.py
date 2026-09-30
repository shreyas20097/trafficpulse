#!/usr/bin/env python3
"""
TrafficPulse premium dark-tech intersection renderer.

Replaces generic SUMO screenshots / flat grey crosses with the
title-slide visual language:
  dark rounded container + luminous road geometry + subtle gradient
  + glowing vehicle dots.

Design tokens (from brief):
  Background      #07131F  (deep navy / blue-black)
  Road gradient   #607789 / #526A7B -> #354A5B
  Centre glow     cyan, very subtle
  Vehicles        #08B9F2 primary, #00E0FF secondary, #F39A3D heavy
  Dots            5-9px equivalent on laptop, enlarged for projector

Outputs (projector-safe, 2400x1500, high contrast):
  artifacts/figures/trafficpulse/*.png

Also writes:
  sumo/trafficpulse/trafficpulse.view.xml  (SUMO-GUI dark preset)

Usage:
  python3 render_trafficpulse_graphics.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# ------------------------------------------------------------------
# Tokens
# ------------------------------------------------------------------
BG = "#07131F"
PANEL = "#0A2136"
PANEL_EDGE = "#1D4E6E"
ROAD_DARK = "#354A5B"
ROAD_MID = "#526A7B"
ROAD_LIGHT = "#607789"
CYAN = "#00E0FF"
CYAN2 = "#08B9F2"
AMBER = "#F39A3D"
LABEL = "#E8F4FA"

OUT = Path("artifacts/figures/trafficpulse")
OUT.mkdir(parents=True, exist_ok=True)

W = 15.0      # road width (both directions)
L = 25.0      # arm half-length (fits inside rounded panel for projector)
JR = 7.0      # junction half-size (slightly brighter square)

rng_master = np.random.default_rng(7)


# ------------------------------------------------------------------
# Canvas
# ------------------------------------------------------------------
def new_canvas():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-36, 36)
    ax.set_aspect("equal")
    ax.axis("off")
    # outer rounded panel (capsule) — road must stay fully inside it
    panel = FancyBboxPatch(
        (-46, -28), 92, 56,
        boxstyle="round,pad=0.4,rounding_size=5",
        facecolor=PANEL, edgecolor=PANEL_EDGE,
        linewidth=1.4, alpha=1.0, zorder=0,
    )
    ax.add_patch(panel)
    # very subtle radial wash behind junction (dark navy -> slightly lighter)
    for r, a in [(17, 0.05), (13, 0.06), (9, 0.07)]:
        ax.add_patch(Circle((0, 0), r, color="#12344D", alpha=a, zorder=1,
                            linewidth=0))
    return fig, ax


def draw_road(ax, lanes_mode="base"):
    """Luminous cross: glow underlay + dark base + mid + bright centre."""
    # cyan atmospheric glow (restrained: 2 passes, low alpha)
    for pad, alpha in [(1.6, 0.05), (0.8, 0.08)]:
        for (x, y, w, h) in [
            (-W / 2 - pad, -L - pad, W + 2 * pad, 2 * L + 2 * pad),  # vertical
            (-L - pad, -W / 2 - pad, 2 * L + 2 * pad, W + 2 * pad),  # horizontal
        ]:
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.1,rounding_size=1.8",
                facecolor=CYAN, edgecolor="none", alpha=alpha, zorder=2))
    # base dark
    for (x, y, w, h) in [
        (-W / 2, -L, W, 2 * L),
        (-L, -W / 2, 2 * L, W),
    ]:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=1.6",
            facecolor=ROAD_DARK, edgecolor="none", zorder=3))
    # mid inset (creates edge falloff -> gradient feel)
    inset = 0.9
    for (x, y, w, h) in [
        (-W / 2 + inset, -L + 0.6, W - 2 * inset, 2 * L - 1.2),
        (-L + 0.6, -W / 2 + inset, 2 * L - 1.2, W - 2 * inset),
    ]:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=1.2",
            facecolor=ROAD_MID, edgecolor="none", alpha=0.96, zorder=4))
    # brighter junction square
    ax.add_patch(FancyBboxPatch(
        (-JR, -JR), 2 * JR, 2 * JR,
        boxstyle="round,pad=0.1,rounding_size=1.4",
        facecolor=ROAD_LIGHT, edgecolor="none", alpha=0.88, zorder=5))
    # faint cyan centre wash (very restrained for cheap projectors)
    ax.add_patch(Circle((0, 0), 6.0, color=CYAN, alpha=0.05, zorder=6,
                        linewidth=0))
    ax.add_patch(Circle((0, 0), 3.5, color="#9BEFFF", alpha=0.05, zorder=6,
                        linewidth=0))
    # central signal node: subtle cyan glow, NOT a traffic-light icon
    for r, a in [(3.2, 0.10), (1.8, 0.28), (0.9, 0.9)]:
        ax.add_patch(Circle((0, 0), r, color=CYAN, alpha=a, zorder=7,
                            linewidth=0))
    ax.add_patch(Circle((0, 0), 0.35, color="#EAFBFF", alpha=0.95, zorder=8,
                        linewidth=0))

    draw_lane_lines(ax, lanes_mode)
    # compass labels — inside dark negative space, never clipped
    for txt, pos in [("N", (0, 26.7)), ("S", (0, -26.7)),
                     ("E", (42.6, 0)), ("W", (-42.6, 0))]:
        ax.text(pos[0], pos[1], txt, color=LABEL, fontsize=13,
                fontweight="bold", ha="center", va="center", zorder=9)


def draw_lane_lines(ax, lanes_mode):
    """Thin, low-opacity separators. Projector-safe: lw>=1.2, no hairlines."""
    lw = 1.3
    ls = (0, (5, 4))
    alpha = 0.20
    col = "#D6EAF3"
    # offsets depend on capacity mode
    if lanes_mode == "onelane":
        # 1 lane per direction -> single centre dashed line per arm
        seps = [0.0]
    elif lanes_mode == "twolane":
        # 2 lanes per direction -> centre + 2 dashed
        seps = [-3.5, 0.0, 3.5]
    else:
        # base visual: 2-3 lanes suggested, keep to 1-2 lines
        seps = [-1.75, 1.75] if lanes_mode == "wide" else [0.0]

    for off in seps:
        a = 0.30 if off == 0.0 else alpha
        # N arm (vertical)
        ax.plot([off, off], [JR - 1, L - 1], color=col, lw=lw,
                ls=ls if off != 0 else (0, (6, 3)), alpha=a, zorder=6)
        # S arm
        ax.plot([off, off], [-L + 1, -JR + 1], color=col, lw=lw,
                ls=ls if off != 0 else (0, (6, 3)), alpha=a, zorder=6)
        # E arm (horizontal)
        ax.plot([JR - 1, L - 1], [off, off], color=col, lw=lw,
                ls=ls if off != 0 else (0, (6, 3)), alpha=a, zorder=6)
        # W arm
        ax.plot([-L + 1, -JR + 1], [off, off], color=col, lw=lw,
                ls=ls if off != 0 else (0, (6, 3)), alpha=a, zorder=6)


# ------------------------------------------------------------------
# Vehicles
# ------------------------------------------------------------------
def arm_positions(n, lane_offsets, dist_range, seed, queue_bias=True):
    """Distances from centre: dense near stop line (queue), sparse outward."""
    rng = np.random.default_rng(seed)
    lo, hi = dist_range  # e.g. (JR+1.5, L-2)
    if n == 0:
        return []
    # queue-biased: triangular/exponential toward stop line
    if queue_bias:
        u = rng.random(n)
        # skew toward lo (near junction): d = lo + (hi-lo)*u^1.8 + jitter
        d = lo + (hi - lo) * np.power(u, 1.8)
        d = np.sort(d)  # nearest junction first
        # enforce min spacing so dots don't overlap (projector readability)
        # dense arms get tighter spacing instead of piling at the far end
        span = hi - lo
        mind = min(2.5, span / max(n, 1) * 0.9) if n > 4 else 2.5
        mind = max(mind, 1.4)
        for i in range(1, len(d)):
            if d[i] < d[i - 1] + mind:
                d[i] = d[i - 1] + mind
        # drop overflow past the road end instead of stacking at the edge
        d = d[d <= hi]
    else:
        d = np.sort(rng.uniform(lo, hi, n))
    m = len(d)
    # round-robin over lanes (balanced columns), then shuffled so the
    # queue order still looks natural instead of perfectly alternating
    lanes = np.arange(m) % len(lane_offsets)
    rng.shuffle(lanes)
    # slight lateral jitter, stays inside road
    jit = rng.normal(0, 0.35, m)
    out = []
    for i in range(m):
        out.append((lane_offsets[lanes[i]] + jit[i], float(d[i])))
    return out


def scatter_with_glow(ax, xs, ys, colors, sizes):
    xs = np.asarray(xs); ys = np.asarray(ys)
    # glow underlay (1 soft pass, restrained for projector)
    ax.scatter(xs, ys, s=np.asarray(sizes) * 2.0, c=colors, alpha=0.15,
               linewidths=0, zorder=10)
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.98, zorder=11,
               edgecolors="white", linewidths=0.9)


def draw_vehicles(ax, per_arm, heavy_share=0.0, seed=11,
                  lanes_mode="base", trails=True, inflow=False):
    """
    per_arm: dict N/S/E/W -> total dots on ENTRY side (toward junction).
    heavy_share: fraction amber.
    lanes_mode: base | onelane | twolane | wide
    inflow: if True, add a few dots outside arms drifting inward
            (forecast->SUMO narrative) + tiny lane arrowheads.
    """
    rng = np.random.default_rng(seed)
    all_x, all_y, all_c, all_s = [], [], [], []
    trail_segs = []

    for arm, n_entry in per_arm.items():
        # auto-split dense arms over 2 entry lanes so peak hours
        # read as two queue columns instead of one overlapping stack
        # (projector readability; geometry stays symmetrical)
        if lanes_mode == "twolane":
            entry_offsets = [1.0, 4.4]
            exit_offsets = [-1.0, -4.4]
        elif lanes_mode == "onelane":
            entry_offsets = [1.9]
            exit_offsets = [-1.9]
        elif lanes_mode == "base" and n_entry > 7:
            entry_offsets = [0.6, 4.0]
            exit_offsets = [-2.0]
        else:
            entry_offsets = [2.0]
            exit_offsets = [-2.0]
        n_exit = max(0, int(round(n_entry * 0.25)))
        pts_entry = arm_positions(n_entry, entry_offsets,
                                  (JR + 1.2, L - 2.0), seed + hash(arm) % 997)
        pts_exit = arm_positions(n_exit, exit_offsets,
                                 (JR + 2.5, L - 3.0),
                                 seed + 500 + hash(arm) % 997,
                                 queue_bias=False)
        for pts, entering in [(pts_entry, True), (pts_exit, False)]:
            for (lat, d) in pts:
                if arm == "N":
                    x, y = lat, d
                elif arm == "S":
                    x, y = -lat, -d
                elif arm == "E":
                    x, y = d, -lat
                else:
                    x, y = -d, lat
                # class: heavy = amber + larger (projector-readable, not huge)
                is_heavy = rng.random() < heavy_share
                if is_heavy:
                    c = AMBER
                    s = rng.choice([170, 210])  # projector: visibly larger
                else:
                    c = CYAN2 if rng.random() < 0.6 else CYAN
                    s = rng.choice([90, 130, 170])  # 3 sizes, depth cue
                all_x.append(x); all_y.append(y); all_c.append(c); all_s.append(s)
                # restrained motion trail: short, only well inside the road
                # so nothing sticks out past the road ends on the projector
                if (trails and entering and not is_heavy
                        and rng.random() < 0.18 and d < L - 4.5):
                    dx, dy = 0, 0
                    tl = 1.6
                    if arm == "N": dy = +tl
                    elif arm == "S": dy = -tl
                    elif arm == "E": dx = +tl
                    else: dx = -tl
                    trail_segs.append(((x, y), (x + dx, y + dy), c))

    scatter_with_glow(ax, all_x, all_y, all_c, all_s)
    for (x0, y0), (x1, y1), c in trail_segs:
        ax.plot([x0, x1], [y0, y1], color=c, alpha=0.28, lw=2.2,
                solid_capstyle="round", zorder=9)

    # small clean lane arrowheads (never crossing the diagram)
    for arm in ["N", "S", "E", "W"]:
        if arm == "N":
            ax.annotate("", xy=(2.0, 11), xytext=(2.0, 13.5),
                        arrowprops=dict(arrowstyle="-|>", color="white",
                                        alpha=0.35, lw=1.4,
                                        mutation_scale=9), zorder=9)
        elif arm == "S":
            ax.annotate("", xy=(-2.0, -11), xytext=(-2.0, -13.5),
                        arrowprops=dict(arrowstyle="-|>", color="white",
                                        alpha=0.35, lw=1.4,
                                        mutation_scale=9), zorder=9)
        elif arm == "E":
            ax.annotate("", xy=(11, -2.0), xytext=(13.5, -2.0),
                        arrowprops=dict(arrowstyle="-|>", color="white",
                                        alpha=0.35, lw=1.4,
                                        mutation_scale=9), zorder=9)
        else:
            ax.annotate("", xy=(-11, 2.0), xytext=(-13.5, 2.0),
                        arrowprops=dict(arrowstyle="-|>", color="white",
                                        alpha=0.35, lw=1.4,
                                        mutation_scale=9), zorder=9)

    if inflow:
        # faint dots drifting in from panel edges (forecast blocks -> road)
        for sx, sy, ex, ey in [(-40, 8, -L - 1, 8), (40, -8, L + 1, -8),
                               (8, 26, 8, L + 1), (-8, -26, -8, -L - 1)]:
            for t in [0.25, 0.5, 0.75]:
                x = sx + (ex - sx) * t + rng.normal(0, 0.5)
                y = sy + (ey - sy) * t + rng.normal(0, 0.5)
                ax.add_patch(Circle((x, y), 0.55, color=CYAN, alpha=0.30,
                                    zorder=8, linewidth=0))


def save(fig, name, title=None):
    if title:
        fig.text(0.5, 0.90, title, color=LABEL, fontsize=20,
                 fontweight="bold", ha="center", va="center")
        fig.text(0.5, 0.06,
                 "TrafficPulse  •  microscopic simulation  •  top-down",
                 color="#7FA3B8", fontsize=9, ha="center", va="center")
    p = OUT / name
    fig.savefig(p, facecolor=BG)  # fixed extent: no bbox_inches shift/clip
    plt.close(fig)
    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")
    return p


# ------------------------------------------------------------------
# Slide set — the same intersection evolving through the deck
# ------------------------------------------------------------------
def main():
    # 1 prototype: clean, sparse cyan dots
    fig, ax = new_canvas()
    draw_road(ax, "base")
    draw_vehicles(ax, {"N": 4, "S": 4, "E": 6, "W": 6},
                  heavy_share=0.0, seed=11)
    save(fig, "01_prototype_sparse.png", "Prototype — clean intersection")

    # 2 forecast -> SUMO: same intersection, inflow from edges
    fig, ax = new_canvas()
    draw_road(ax, "base")
    draw_vehicles(ax, {"N": 6, "S": 6, "E": 8, "W": 8},
                  heavy_share=0.05, seed=22, inflow=True)
    save(fig, "02_forecast_to_sumo.png", "Forecast → SUMO — demand flows in")

    # 3 demand ladder: low -> base -> high -> peak (density only)
    for tag, title, n, seed in [
        ("low", "Demand — low", {"N": 3, "S": 3, "E": 4, "W": 4}, 31),
        ("base", "Demand — base", {"N": 6, "S": 6, "E": 9, "W": 9}, 32),
        ("high", "Demand — high", {"N": 10, "S": 10, "E": 14, "W": 14}, 33),
        ("peak", "Demand — peak", {"N": 14, "S": 14, "E": 19, "W": 19}, 34),
    ]:
        fig, ax = new_canvas()
        draw_road(ax, "base")
        draw_vehicles(ax, n, heavy_share=0.05, seed=seed)
        save(fig, f"03_demand_{tag}.png", title)

    # 4 composition: identical totals, cyan -> amber swap
    tot = {"N": 8, "S": 8, "E": 11, "W": 11}
    fig, ax = new_canvas()
    draw_road(ax, "base")
    draw_vehicles(ax, tot, heavy_share=0.0, seed=41)
    save(fig, "04a_composition_all_cars.png",
         "Composition — all cars (reference)")
    fig, ax = new_canvas()
    draw_road(ax, "base")
    draw_vehicles(ax, tot, heavy_share=0.18, seed=41)  # same seed: same slots
    save(fig, "04b_composition_heavy_mix.png",
         "Composition — heavy mix (same total)")

    # 5 capacity: one-lane vs two-lane approaches
    cap = {"N": 9, "S": 9, "E": 12, "W": 12}
    fig, ax = new_canvas()
    draw_road(ax, "onelane")
    draw_vehicles(ax, cap, heavy_share=0.05, seed=55, lanes_mode="onelane")
    save(fig, "05a_capacity_one_lane.png", "Capacity — one lane / approach")
    fig, ax = new_canvas()
    draw_road(ax, "twolane")
    draw_vehicles(ax, cap, heavy_share=0.05, seed=55, lanes_mode="twolane")
    save(fig, "05b_capacity_two_lanes.png", "Capacity — two lanes / approach")

    # 6 conclusion: return to clean
    fig, ax = new_canvas()
    draw_road(ax, "base")
    draw_vehicles(ax, {"N": 4, "S": 4, "E": 6, "W": 6},
                  heavy_share=0.0, seed=11)
    save(fig, "06_conclusion_clean.png", "Conclusion — the same intersection")

    # hero (no title, for title slide reuse)
    fig, ax = new_canvas()
    draw_road(ax, "base")
    draw_vehicles(ax, {"N": 5, "S": 5, "E": 7, "W": 7},
                  heavy_share=0.06, seed=11)
    save(fig, "trafficpulse_intersection_hero.png")

    write_view_preset()
    print("\nAll graphics in artifacts/figures/trafficpulse/")


def write_view_preset():
    """SUMO-GUI dark preset so live screenshots stay on-brand."""
    xml = """<viewsettings>
    <!-- TrafficPulse dark-tech preset. Load in SUMO-GUI: View Settings > Load. -->
    <scheme name="TrafficPulseDark">
        <background backgroundColor="6,17,28" showGrid="0" gridXSize="100" gridYSize="100"/>
        <edges laneColor="82,106,123" laneColor2="96,119,137" showBikeMarkings="0" showLinkRules="0" spreadSuperposed="0" widthExaggeration="1.6" scaleWidth="1"/>
        <vehicles vehicleColor="8,185,242" vehicleColor2="0,224,255" showBlinker="0" showMinGap="0" vehicleExaggeration="3.0" vehicleScaleWidth="1"/>
        <junctions junctionColor="0,224,255" drawJunctionShape="1" junctionSize="6"/>
        <persons personColor="243,154,61" personExaggeration="2.0"/>
        <decals showDecals="0"/>
    </scheme>
</viewsettings>
"""
    p = Path("sumo/trafficpulse/trafficpulse.view.xml")
    p.write_text(xml)
    print(f"wrote {p}")

    colors = """<routes>
    <!-- TrafficPulse brand vehicle colours (SUMO-native RGB).
         cars: cyan/blue dots  |  heavy: amber dots. Exaggerate size for projector. -->
    <vType id="short"      color="8,185,242"   guiShape="passenger"/>
    <vType id="towing"     color="0,224,255"   guiShape="passenger"/>
    <vType id="rigid"      color="8,185,242"   guiShape="truck"/>
    <vType id="heavy"      color="243,154,61"  guiShape="truck"/>
    <vType id="motorcycle" color="0,224,255"   guiShape="motorcycle"/>
</routes>
"""
    p2 = Path("sumo/trafficpulse/trafficpulse_vtype_colors.xml")
    p2.write_text(colors)
    print(f"wrote {p2}  (reference: add color= to vTypes for live GUI)")


if __name__ == "__main__":
    main()
