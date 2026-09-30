#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/omarchy/Projects/de"
SCRIPT="$PROJECT/trafficpulse_dynamic.py"

cd "$PROJECT"

echo "== TrafficPulse dynamic-controller fix v2 =="
echo "Project: $PROJECT"
echo

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: $SCRIPT does not exist."
    exit 1
fi

# ------------------------------------------------------------
# Backup
# ------------------------------------------------------------
BACKUP="$SCRIPT.bak.$(date +%Y%m%d_%H%M%S)"
cp -- "$SCRIPT" "$BACKUP"
echo "Backup created:"
echo "  $BACKUP"
echo

# ------------------------------------------------------------
# Patch using Python.
#
# We do NOT depend on guessed comments such as "# Build routes".
# Instead we:
#
#   1. Ensure PROFILE_FILE exists.
#   2. Locate the old calibration section using its actual
#      variable names.
#   3. Find the first later block that starts constructing the
#      demand XML / route data.
#   4. Replace only the calibration section.
#
# If the structure cannot be identified safely, the original
# controller remains untouched and the script exits.
# ------------------------------------------------------------
"$PROJECT/.venv/bin/python" - "$SCRIPT" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()

original = text

# ------------------------------------------------------------
# Ensure PROFILE_FILE exists
# ------------------------------------------------------------
if "PROFILE_FILE" not in text:
    anchor = 'COMPOSITION_FILE=PROJECT/"artifacts/tirtl_vehicle_composition_by_hour.csv"'

    if anchor not in text:
        raise SystemExit(
            "ERROR: Could not find COMPOSITION_FILE definition."
        )

    text = text.replace(
        anchor,
        anchor
        + '\n'
        + 'PROFILE_FILE=PROJECT/"artifacts/tirtl_representative_15min_profile.csv"',
        1,
    )

# ------------------------------------------------------------
# Find the calibration area.
#
# The old implementation uses source_details and computes a
# calibrated/dynamic total from the four independent streams.
#
# Locate the FIRST assignment involving calibrated demand, then
# walk backwards to include its associated profile/calibration
# setup where appropriate.
# ------------------------------------------------------------
calibration_patterns = [
    r"^\s*calibrated_total\s*=",
    r"^\s*calibrated\s*=",
    r"^\s*calibrated_demand\s*=",
    r"^\s*dynamic_total\s*=",
    r"^\s*calibrated_direction",
]

lines = text.splitlines(keepends=True)

candidate_lines = []
for i, line in enumerate(lines):
    if any(re.search(p, line) for p in calibration_patterns):
        candidate_lines.append(i)

if not candidate_lines:
    raise SystemExit(
        "ERROR: Could not find the old demand-calibration code. "
        "No modification was made."
    )

start = min(candidate_lines)

# Walk upward to include preceding calibration setup.
# Stop at a clear structural boundary:
#   - beginning of a function
#   - blank/comment boundary after substantial code
#   - previous top-level statement
#
# In this controller the calibration happens inside main(), so we
# retain the indentation-based structure.
indent = len(lines[start]) - len(lines[start].lstrip())

while start > 0:
    prev = lines[start - 1]
    stripped = prev.strip()

    prev_indent = len(prev) - len(prev.lstrip())

    # Don't cross a function definition.
    if stripped.startswith(("def ", "async def ")):
        break

    # Don't climb into unrelated top-level code.
    if stripped and prev_indent < indent and not stripped.startswith("#"):
        break

    # Include nearby setup/comments belonging to calibration.
    if stripped == "" and start > 1:
        # Include one blank line only if previous lines look like
        # calibration setup.
        joined = "".join(lines[max(0, start-8):start])
        if not any(
            x in joined
            for x in (
                "source_details",
                "reference",
                "calibrat",
                "DIRECTION_SHARE",
            )
        ):
            break

    start -= 1

# ------------------------------------------------------------
# Find where the calibration ends.
#
# We look for statements that clearly consume the calculated
# direction counts to generate SUMO demand.
# ------------------------------------------------------------
end = None

end_patterns = [
    r"^\s*routes?\s*=",
    r"^\s*route_xml\s*=",
    r"^\s*flow_xml\s*=",
    r"^\s*xml\s*=",
    r"^\s*demand_xml\s*=",
    r"^\s*route_file\s*=",
    r"^\s*route_path\s*=",
    r"^\s*write_text\s*\(",
    r"^\s*ET\.Element\s*\(",
    r"^\s*root\s*=\s*ET\.Element",
]

# Don't choose something before the calibration start.
for i in range(start + 1, len(lines)):
    line = lines[i]
    if any(re.search(p, line) for p in end_patterns):
        end = i
        break

# If no obvious XML creation marker exists, find the next top-level/
# same-level substantial statement after calibration.
if end is None:
    for i in range(start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            continue

        cur_indent = len(line) - len(line.lstrip())

        if cur_indent <= indent and not stripped.startswith(("#", "@")):
            # Avoid stopping inside obvious calibration continuations.
            if not any(
                token in stripped
                for token in (
                    "source_details",
                    "current_source_total",
                    "predicted_source_total",
                    "forecast_multiplier",
                    "direction_integer",
                    "actual_sim_total",
                    "profile_row",
                )
            ):
                end = i
                break

if end is None or end <= start:
    raise SystemExit(
        "ERROR: Could not safely determine the end of the old "
        "calibration block. No modification was made."
    )

# ------------------------------------------------------------
# Construct replacement with indentation matching old block.
# ------------------------------------------------------------
base_indent = re.match(r"^\s*", lines[start]).group(0)

replacement = f'''{base_indent}# ============================================================
{base_indent}# CORRECTED DEMAND CALIBRATION
{base_indent}#
{base_indent}# The four TIRTL source streams are independent sensors.
{base_indent}# They are NOT four physical arms of one intersection.
{base_indent}#
{base_indent}# Therefore:
{base_indent}#
{base_indent}#   predicted N + predicted S + predicted E + predicted W
{base_indent}#
{base_indent}# must NOT become the hypothetical intersection demand.
{base_indent}#
{base_indent}# Instead:
{base_indent}#
{base_indent}#   representative TIRTL profile demand
{base_indent}#       x
{base_indent}#   XGBoost forecast multiplier
{base_indent}#       =
{base_indent}#   hypothetical intersection demand
{base_indent}#
{base_indent}# The model multiplier is:
{base_indent}#
{base_indent}#   sum(predicted source streams)
{base_indent}#   ---------------------------
{base_indent}#   sum(current source streams)
{base_indent}#
{base_indent}# This preserves the realistic demand scale while allowing
{base_indent}# XGBoost to predict the short-term deviation.
{base_indent}# ============================================================

{base_indent}profile = pd.read_csv(PROFILE_FILE)

{base_indent}profile["hour"] = profile["hour"].astype(int)
{base_indent}profile["time_bin"] = profile["time_bin"].astype(str)

{base_indent}profile_row = profile[
{base_indent}    (profile["hour"] == int(timestamp.hour))
{base_indent}    & (profile["time_bin"] == str(args.time))
{base_indent}]

{base_indent}if profile_row.empty:
{base_indent}    raise ValueError(
{base_indent}        f"No representative TIRTL profile row for "
{base_indent}        f"hour={{timestamp.hour}}, time_bin={{args.time}}"
{base_indent}    )

{base_indent}representative_current = float(
{base_indent}    profile_row.iloc[0]["mean_demand"]
{base_indent})

{base_indent}current_source_total = sum(
{base_indent}    float(source_details[d]["current_observed_total"])
{base_indent}    for d in SOURCE_STREAMS
{base_indent})

{base_indent}predicted_source_total = sum(
{base_indent}    float(source_details[d]["model_prediction"])
{base_indent}    for d in SOURCE_STREAMS
{base_indent})

{base_indent}if current_source_total <= 0:
{base_indent}    raise ValueError(
{base_indent}        "Current TIRTL source total is zero; "
{base_indent}        "cannot calculate forecast multiplier."
{base_indent}    )

{base_indent}forecast_multiplier = (
{base_indent}    predicted_source_total / current_source_total
{base_indent})

{base_indent}# Guard against pathological extrapolation.
{base_indent}forecast_multiplier = max(
{base_indent}    0.50,
{base_indent}    min(1.50, forecast_multiplier),
{base_indent})

{base_indent}dynamic_total = max(
{base_indent}    1,
{base_indent}    int(round(
{base_indent}        representative_current * forecast_multiplier
{base_indent}    ))
{base_indent})

{base_indent}direction_integer = exact_allocate(
{base_indent}    dynamic_total,
{base_indent}    DIRECTION_SHARE
{base_indent})

{base_indent}actual_sim_total = sum(direction_integer.values())

{base_indent}calibration_details = {{
{base_indent}    "representative_profile_demand": representative_current,
{base_indent}    "current_source_total": current_source_total,
{base_indent}    "predicted_source_total": predicted_source_total,
{base_indent}    "forecast_multiplier": forecast_multiplier,
{base_indent}    "dynamic_total": dynamic_total,
{base_indent}    "actual_sim_total": actual_sim_total,
{base_indent}}}

{base_indent}print()
{base_indent}print("Demand calibration:")
{base_indent}print(
{base_indent}    f"  Representative profile : "
{base_indent}    f"{{representative_current:.2f}} veh/15min"
{base_indent})
{base_indent}print(
{base_indent}    f"  Current source total   : "
{base_indent}    f"{{current_source_total:.2f}}"
{base_indent})
{base_indent}print(
{base_indent}    f"  Predicted source total : "
{base_indent}    f"{{predicted_source_total:.2f}}"
{base_indent})
{base_indent}print(
{base_indent}    f"  XGBoost multiplier     : "
{base_indent}    f"{{forecast_multiplier:.4f}}"
{base_indent})
{base_indent}print(
{base_indent}    f"  Dynamic intersection   : "
{base_indent}    f"{{dynamic_total}} veh/15min"
{base_indent})
{base_indent}print(
{base_indent}    f"  N/S/E/W                : "
{base_indent}    f"{{direction_integer}}"
{base_indent})
{base_indent}print(
{base_indent}    f"  Simulation total       : "
{base_indent}    f"{{actual_sim_total}} veh/15min"
{base_indent})

'''

new_lines = lines[:start] + [replacement] + lines[end:]
new_text = "".join(new_lines)

# ------------------------------------------------------------
# Safety checks before writing.
# ------------------------------------------------------------

required = [
    "PROFILE_FILE",
    "representative_current",
    "current_source_total",
    "predicted_source_total",
    "forecast_multiplier",
    "dynamic_total",
    "direction_integer",
    "actual_sim_total",
]

missing = [x for x in required if x not in new_text]

if missing:
    raise SystemExit(
        "ERROR: Safety validation failed; missing: "
        + ", ".join(missing)
        + ". No modification was made."
    )

# Make sure old dangerous expression isn't still present.
dangerous_patterns = [
    r"calibrated_total\s*=\s*sum\s*\(",
    r"dynamic_total\s*=\s*sum\s*\(",
]

for pattern in dangerous_patterns:
    if re.search(pattern, new_text):
        raise SystemExit(
            f"ERROR: Old dangerous calibration expression remains: "
            f"{pattern}"
        )

path.write_text(new_text)

print()
print("Controller successfully patched.")
print(f"  File: {path}")
print(f"  Replaced lines: {start + 1}..{end}")
PY

# ------------------------------------------------------------
# Syntax validation
# ------------------------------------------------------------
echo
echo "== Syntax check =="

"$PROJECT/.venv/bin/python" -m py_compile "$SCRIPT"

echo "PASS: Python syntax is valid."

# ------------------------------------------------------------
# Verify required files
# ------------------------------------------------------------
echo
echo "== Required input check =="

for f in \
    "$PROJECT/artifacts/tirtl_features.csv" \
    "$PROJECT/artifacts/tirtl_training/xgboost.joblib" \
    "$PROJECT/artifacts/tirtl_vehicle_composition_by_hour.csv" \
    "$PROJECT/artifacts/tirtl_representative_15min_profile.csv"
do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing required file:"
        echo "  $f"
        exit 1
    fi
    echo "OK  $f"
done

# ------------------------------------------------------------
# Run the six timestamps WITHOUT running SUMO.
# ------------------------------------------------------------
echo
echo "============================================================"
echo " SIX-TIMESTAMP DEMAND AUDIT"
echo "============================================================"

AUDIT="$PROJECT/artifacts/tirtl_dynamic_audit_fixed.txt"
: > "$AUDIT"

for T in 07:30 08:00 12:00 17:00 17:15 18:00; do
    echo
    echo "---------------- $T ----------------"

    set +e
    "$PROJECT/.venv/bin/python" "$SCRIPT" \
        --time "$T" \
        --no-run 2>&1 | tee -a "$AUDIT"

    status=${PIPESTATUS[0]}
    set -e

    if [[ "$status" -ne 0 ]]; then
        echo
        echo "ERROR: Controller failed at $T."
        echo
        echo "The original controller backup is:"
        echo "  $BACKUP"
        exit "$status"
    fi
done

# ------------------------------------------------------------
# Parse output and reject old 1500-2000 vehicle behavior.
# ------------------------------------------------------------
echo
echo "============================================================"
echo " DEMAND-SCALE VALIDATION"
echo "============================================================"

"$PROJECT/.venv/bin/python" - "$AUDIT" <<'PY'
from pathlib import Path
import re
import sys

log = Path(sys.argv[1]).read_text()

matches = re.findall(
    r"Dynamic intersection\s*:\s*([0-9]+)\s*veh/15min",
    log,
)

if len(matches) != 6:
    print(
        f"ERROR: Expected 6 dynamic-demand values, "
        f"found {len(matches)}."
    )
    sys.exit(1)

values = list(map(int, matches))

times = [
    "07:30",
    "08:00",
    "12:00",
    "17:00",
    "17:15",
    "18:00",
]

for t, v in zip(times, values):
    print(f"{t:>5}  {v:4d} veh/15min")

print()

# The previous bug produced approximately:
#   1558, 1525, 1544, 1995, 1806, 1555
#
# The corrected approach should be roughly around the
# representative profile scale, not the sum of four freeway
# sensors.
if any(v >= 1400 for v in values):
    print(
        "FAIL: At least one demand value still looks like "
        "the old four-stream sum."
    )
    sys.exit(2)

if any(v < 300 for v in values):
    print(
        "FAIL: At least one demand value unexpectedly collapsed "
        "below 300 veh/15min."
    )
    sys.exit(3)

print("PASS: Demand is in the expected single-intersection scale.")
PY

# ------------------------------------------------------------
# Run the real simulation for 17:15.
# ------------------------------------------------------------
echo
echo "============================================================"
echo " REAL SUMO VERIFICATION — 17:15"
echo "============================================================"

"$PROJECT/.venv/bin/python" "$SCRIPT" --time 17:15

echo
echo "============================================================"
echo " DONE"
echo "============================================================"
echo
echo "Controller:"
echo "  $SCRIPT"
echo
echo "Backup:"
echo "  $BACKUP"
echo
echo "Audit:"
echo "  $AUDIT"
echo
echo "The corrected pipeline is:"
echo
echo "  TIRTL representative profile"
echo "             ↓"
echo "    realistic intersection scale"
echo "             ×"
echo "    XGBoost forecast multiplier"
echo "             ↓"
echo "    predicted intersection demand"
echo "             ↓"
echo "       exact N/S/E/W split"
echo "             ↓"
echo "            SUMO"
