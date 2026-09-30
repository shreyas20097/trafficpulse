from pathlib import Path
import subprocess
import sys


# ============================================================
# Paths
# ============================================================

BASE = Path("sumo/trafficpulse").resolve()

NETWORK = BASE / "network"
DEMAND = BASE / "demand"
OUTPUT = BASE / "output"

for directory in (NETWORK, DEMAND, OUTPUT):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# TrafficPulse baseline assumptions
# ============================================================

TOTAL_VEHICLES_15MIN = 1050

# Vehicles are released during the first 15 minutes.
DEMAND_DURATION = 900

# Simulation continues after the final departure so vehicles
# can clear the intersection.
SIM_END = 3600

SEED = 42

# Derived from the TIRTL directional analysis.
DIRECTION_SHARE = {
    "N": 0.2061,
    "S": 0.2034,
    "E": 0.2934,
    "W": 0.2971,
}

# TIRTL peak-period vehicle composition.
VEHICLE_SHARE = {
    "short": 0.895,
    "rigid": 0.074,
    "heavy": 0.016,
    "towing": 0.009,
    "motorcycle": 0.005,
}

# This is explicitly a hypothetical-intersection assumption.
# TIRTL does not directly provide turn proportions.
TURN_SHARE = {
    "left": 0.20,
    "straight": 0.70,
    "right": 0.10,
}

# 50 km/h.
SPEED = 13.89


# ============================================================
# Helpers
# ============================================================

def allocate_exact(total: int, shares: dict[str, float]) -> dict[str, int]:
    """
    Allocate an integer total according to fractional shares
    while guaranteeing that the returned values sum exactly
    to total.
    """
    keys = list(shares)

    raw = {
        key: total * shares[key]
        for key in keys
    }

    result = {
        key: int(raw[key])
        for key in keys
    }

    remaining = total - sum(result.values())

    # Largest-remainder method.
    fractions = sorted(
        keys,
        key=lambda key: raw[key] - result[key],
        reverse=True,
    )

    for key in fractions[:remaining]:
        result[key] += 1

    return result


def require_sum(
    name: str,
    values: dict[str, int],
    expected: int,
) -> None:
    actual = sum(values.values())

    if actual != expected:
        raise RuntimeError(
            f"{name} allocation error: "
            f"expected {expected}, got {actual}"
        )


# ============================================================
# Build network input
# ============================================================

nodes_xml = """<nodes>
    <node id="C" x="0" y="0" type="traffic_light"/>

    <node id="N" x="0" y="250" type="priority"/>
    <node id="S" x="0" y="-250" type="priority"/>
    <node id="E" x="250" y="0" type="priority"/>
    <node id="W" x="-250" y="0" type="priority"/>
</nodes>
"""

edges_xml = f"""<edges>
    <edge id="N_C" from="N" to="C" numLanes="1" speed="{SPEED}"/>
    <edge id="C_N" from="C" to="N" numLanes="1" speed="{SPEED}"/>

    <edge id="S_C" from="S" to="C" numLanes="1" speed="{SPEED}"/>
    <edge id="C_S" from="C" to="S" numLanes="1" speed="{SPEED}"/>

    <edge id="E_C" from="E" to="C" numLanes="1" speed="{SPEED}"/>
    <edge id="C_E" from="C" to="E" numLanes="1" speed="{SPEED}"/>

    <edge id="W_C" from="W" to="C" numLanes="1" speed="{SPEED}"/>
    <edge id="C_W" from="C" to="W" numLanes="1" speed="{SPEED}"/>
</edges>
"""

(NETWORK / "intersection.nod.xml").write_text(nodes_xml)
(NETWORK / "intersection.edg.xml").write_text(edges_xml)

net_file = NETWORK / "intersection.net.xml"


# ============================================================
# Build SUMO network
# ============================================================

netconvert = str(
    Path(".sumo-venv/bin/netconvert").resolve()
)

cmd = [
    netconvert,

    "--node-files",
    str(NETWORK / "intersection.nod.xml"),

    "--edge-files",
    str(NETWORK / "intersection.edg.xml"),

    "--output-file",
    str(net_file),

    "--tls.default-type",
    "static",

    "--tls.layout",
    "opposites",

    # Two opposing signal groups:
    # 30 s green + 3 s yellow each = 66 s cycle.
    "--tls.cycle.time",
    "66",

    "--tls.yellow.time",
    "3",

    # No protected-left extra phase in baseline.
    "--tls.left-green.time",
    "0",
]

print("=" * 80)
print("BUILDING TRAFFICPULSE SUMO NETWORK")
print("=" * 80)

result = subprocess.run(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

print(result.stdout)

if result.returncode != 0:
    raise RuntimeError("netconvert failed")

if not net_file.exists():
    raise RuntimeError(
        f"netconvert reported success but {net_file} does not exist"
    )

print(f"Network written to: {net_file}")


# ============================================================
# Vehicle definitions
# ============================================================

routes_xml = """<routes>

    <!--
        TIRTL-derived vehicle groups.

        These are simulation categories, not claims that TIRTL
        separately identifies every bus/truck subtype.
    -->

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


# ============================================================
# Movement mapping
#
# Incoming direction:
#   N -> southbound
#   S -> northbound
#   E -> westbound
#   W -> eastbound
# ============================================================

movements = {
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


# ============================================================
# Exact directional demand
# ============================================================

direction_counts = allocate_exact(
    TOTAL_VEHICLES_15MIN,
    DIRECTION_SHARE,
)

require_sum(
    "Directional",
    direction_counts,
    TOTAL_VEHICLES_15MIN,
)


# ============================================================
# Exact vehicle composition for each approach
# ============================================================

vehicle_counts = {
    direction: allocate_exact(
        direction_counts[direction],
        VEHICLE_SHARE,
    )
    for direction in direction_counts
}

for direction in direction_counts:
    require_sum(
        f"Vehicle composition {direction}",
        vehicle_counts[direction],
        direction_counts[direction],
    )


# ============================================================
# Exact turn allocation for each approach
# ============================================================

movement_counts = {
    direction: allocate_exact(
        direction_counts[direction],
        TURN_SHARE,
    )
    for direction in direction_counts
}

for direction in direction_counts:
    require_sum(
        f"Turn allocation {direction}",
        movement_counts[direction],
        direction_counts[direction],
    )


# ============================================================
# Report demand
# ============================================================

print()
print("=" * 80)
print("BASELINE DEMAND")
print("=" * 80)

print(
    f"Total demand: "
    f"{TOTAL_VEHICLES_15MIN} vehicles / 15 min"
)

print()

for direction in ("N", "S", "E", "W"):
    print(
        f"{direction}: "
        f"{direction_counts[direction]} vehicles"
    )

print()

print("Vehicle composition by approach:")

for direction in ("N", "S", "E", "W"):
    print(
        f"  {direction}: "
        + ", ".join(
            f"{key}={value}"
            for key, value
            in vehicle_counts[direction].items()
        )
    )

print()

print("Turn allocation by approach:")

for direction in ("N", "S", "E", "W"):
    print(
        f"  {direction}: "
        + ", ".join(
            f"{key}={value}"
            for key, value
            in movement_counts[direction].items()
        )
    )


# ============================================================
# Build route definitions ONCE
# ============================================================

flow_lines = []

for direction in ("N", "S", "E", "W"):
    for turn in ("left", "straight", "right"):

        route_id = f"{direction}_{turn}"

        flow_lines.append(
            f"""    <route
        id="{route_id}"
        edges="{movements[direction][turn]}"
    />
"""
        )


# ============================================================
# Build flows
#
# Each route is defined once.
# Each movement gets its own exact vehicle-class allocation.
# ============================================================

flow_id = 0

for direction in ("N", "S", "E", "W"):

    for turn in ("left", "straight", "right"):

        movement_total = movement_counts[direction][turn]

        movement_vehicle_counts = allocate_exact(
            movement_total,
            VEHICLE_SHARE,
        )

        require_sum(
            f"{direction}_{turn} vehicle classes",
            movement_vehicle_counts,
            movement_total,
        )

        route_id = f"{direction}_{turn}"

        for vehicle_type, number in movement_vehicle_counts.items():

            if number <= 0:
                continue

            flow_lines.append(
                f"""    <flow
        id="f_{flow_id}"
        type="{vehicle_type}"
        route="{route_id}"
        begin="0"
        end="{DEMAND_DURATION}"
        number="{number}"
        departLane="best"
        departSpeed="max"
    />
"""
            )

            flow_id += 1


routes_xml += "\n".join(flow_lines)
routes_xml += "\n</routes>\n"

route_file = DEMAND / "baseline.rou.xml"
route_file.write_text(routes_xml)


# ============================================================
# Validate generated total from <flow number="...">
# ============================================================

generated_total = 0

for line in routes_xml.splitlines():
    stripped = line.strip()

    if stripped.startswith("number="):
        generated_total += int(
            stripped.split('"')[1]
        )

if generated_total != TOTAL_VEHICLES_15MIN:
    raise RuntimeError(
        "Generated flow total mismatch: "
        f"expected {TOTAL_VEHICLES_15MIN}, "
        f"got {generated_total}"
    )

print()
print(
    f"Route file written to: {route_file}"
)
print(
    f"Generated vehicle total: {generated_total}"
)


# ============================================================
# SUMO configuration
# ============================================================

tripinfo_file = OUTPUT / "tripinfo.xml"

sumocfg = f"""<configuration>

    <input>
        <net-file value="network/intersection.net.xml"/>
        <route-files value="demand/baseline.rou.xml"/>
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
        <tripinfo-output value="output/tripinfo.xml"/>
        <tripinfo-output.write-unfinished value="true"/>
    </output>

    <report>
        <duration-log.statistics value="true"/>
    </report>

    <random_number>
        <seed value="{SEED}"/>
    </random_number>

</configuration>
"""

cfg_file = BASE / "trafficpulse_baseline.sumocfg"
cfg_file.write_text(sumocfg)


print(
    f"Config written to: {cfg_file}"
)

print()
print("=" * 80)
print("BUILD COMPLETE")
print("=" * 80)

print()
print("Network:")
print(net_file)

print("Demand:")
print(route_file)

print("Config:")
print(cfg_file)

print()
print("Run headless:")
print(
    f".sumo-venv/bin/sumo "
    f"-c {cfg_file}"
)

print()
print("Run GUI:")
print(
    f"LIBGL_ALWAYS_SOFTWARE=1 "
    f".sumo-venv/bin/sumo-gui "
    f"-c {cfg_file} "
    f"2>/dev/null"
)
