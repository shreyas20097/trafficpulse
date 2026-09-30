from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("tirtl_15min_volume_classification_january_2026")
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

RAW_OUT = OUT / "tirtl_traffic_15min.csv"
FEATURE_OUT = OUT / "tirtl_features.csv"

# ------------------------------------------------------------
# Official TIRTL / Austroads Level 3 class groups
# ------------------------------------------------------------

# 0 = Motorcycles
# 1 = Short vehicle
# 2 = Short vehicle towing
# 3-5 = rigid truck / bus categories
# 6-12 = articulated / heavy vehicle combinations
# 13 = Error
# 14 = Bicycles

CLASS_MAP = {
    0: "motorcycle",
    1: "short_vehicle",
    2: "short_vehicle_towing",
    3: "rigid_truck_bus",
    4: "rigid_truck_bus",
    5: "rigid_truck_bus",
    6: "heavy_articulated",
    7: "heavy_articulated",
    8: "heavy_articulated",
    9: "heavy_articulated",
    10: "heavy_articulated",
    11: "heavy_articulated",
    12: "heavy_articulated",
    13: "error",
    14: "bicycle",
}

USECOLS = [
    "date",
    "time_bin",
    "site",
    "heading",
    "vehicle_class",
    "speed_bin",
    "volume",
]


def speed_bin_midpoint(x):
    """
    Convert speed-bin text to an approximate midpoint.

    Examples:
        0km/hr to < 5km/hr -> 2.5
        95km/hr to < 100km/hr -> 97.5
        150km/hr + -> 155
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s.endswith("km/hr +"):
        try:
            return float(s.split("km/hr")[0]) + 2.5
        except ValueError:
            return np.nan

    try:
        left = s.split("km/hr")[0]
        right = s.split("to <")[1].split("km/hr")[0]

        return (
            float(left.strip()) +
            float(right.strip())
        ) / 2.0

    except (ValueError, IndexError):
        return np.nan


# ============================================================
# STEP 1: Aggregate raw speed-bin records
# ============================================================

print("=" * 80)
print("TIRTL PREPROCESSING")
print("=" * 80)

files = sorted(ROOT.glob("*.csv"))

if len(files) != 31:
    print(f"WARNING: expected 31 daily files, found {len(files)}")

daily = []

for i, path in enumerate(files, 1):

    print(
        f"[{i:02d}/{len(files)}] "
        f"Reading {path.name}"
    )

    df = pd.read_csv(
        path,
        usecols=USECOLS
    )

    # Basic types
    df["site"] = pd.to_numeric(
        df["site"],
        errors="coerce"
    )

    df["vehicle_class"] = pd.to_numeric(
        df["vehicle_class"],
        errors="coerce"
    )

    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce"
    ).fillna(0)

    # Map official class IDs to groups
    df["class_group"] = (
        df["vehicle_class"]
        .map(CLASS_MAP)
        .fillna("unknown")
    )

    # --------------------------------------------------------
    # Vehicle counts:
    #
    # Each raw row is one vehicle class + speed bin within
    # a 15-minute interval.
    #
    # Therefore SUM over speed_bin.
    # --------------------------------------------------------

    counts = (
        df.pivot_table(
            index=[
                "date",
                "time_bin",
                "site",
                "heading",
            ],
            columns="class_group",
            values="volume",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    # Ensure all expected columns exist
    expected_groups = [
        "motorcycle",
        "short_vehicle",
        "short_vehicle_towing",
        "rigid_truck_bus",
        "heavy_articulated",
        "bicycle",
        "error",
        "unknown",
    ]

    for c in expected_groups:
        if c not in counts.columns:
            counts[c] = 0

    # --------------------------------------------------------
    # Speed feature
    #
    # Weighted mean speed across all detected vehicles.
    # --------------------------------------------------------

    df["speed_mid"] = df["speed_bin"].map(
        speed_bin_midpoint
    )

    df["speed_volume"] = (
        df["speed_mid"] * df["volume"]
    )

    speed_stats = (
        df.groupby(
            ["date", "time_bin", "site", "heading"],
            as_index=False
        )
        .agg(
            speed_volume=("speed_volume", "sum"),
            speed_count=("volume", "sum"),
        )
    )

    speed_stats["mean_speed"] = np.where(
        speed_stats["speed_count"] > 0,
        speed_stats["speed_volume"]
        / speed_stats["speed_count"],
        np.nan,
    )

    speed_stats = speed_stats[
        [
            "date",
            "time_bin",
            "site",
            "heading",
            "mean_speed",
        ]
    ]

    counts = counts.merge(
        speed_stats,
        on=[
            "date",
            "time_bin",
            "site",
            "heading",
        ],
        how="left",
    )

    # --------------------------------------------------------
    # Totals
    # --------------------------------------------------------

    motorized_cols = [
        "motorcycle",
        "short_vehicle",
        "short_vehicle_towing",
        "rigid_truck_bus",
        "heavy_articulated",
    ]

    all_vehicle_cols = motorized_cols + ["bicycle"]

    counts["total_motorized"] = counts[
        motorized_cols
    ].sum(axis=1)

    counts["total_all"] = counts[
        all_vehicle_cols
    ].sum(axis=1)

    daily.append(counts)

# Combine the 31 processed days
traffic = pd.concat(
    daily,
    ignore_index=True
)

# Sort
traffic["datetime"] = pd.to_datetime(
    traffic["date"].astype(str)
    + " "
    + traffic["time_bin"].astype(str)
)

traffic = traffic.sort_values(
    ["site", "heading", "datetime"]
).reset_index(drop=True)

# Save aggregated dataset
traffic.to_csv(
    RAW_OUT,
    index=False
)

print()
print("=" * 80)
print("AGGREGATED DATASET")
print("=" * 80)

print(f"Rows:       {len(traffic):,}")
print(f"Sites:      {traffic['site'].nunique()}")
print(f"Headings:   {traffic['heading'].nunique()}")
print(
    f"Date range: "
    f"{traffic['datetime'].min()} → "
    f"{traffic['datetime'].max()}"
)

print(f"\nSaved: {RAW_OUT}")


# ============================================================
# STEP 2: Feature engineering
# ============================================================

print()
print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

df = traffic.copy()

# Calendar features
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["day_of_week"] = df["datetime"].dt.dayofweek
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Time-of-day cyclic features
minutes = (
    df["hour"] * 60
    + df["minute"]
)

df["time_sin"] = np.sin(
    2 * np.pi * minutes / 1440
)

df["time_cos"] = np.cos(
    2 * np.pi * minutes / 1440
)

# ------------------------------------------------------------
# Lag / rolling features
#
# All operations are performed independently for each
# site + heading, preventing information leakage between
# locations/directions.
# ------------------------------------------------------------

group_cols = ["site", "heading"]

for lag in [1, 2, 4, 8]:

    df[f"lag_{lag}"] = (
        df.groupby(group_cols)["total_motorized"]
          .shift(lag)
    )

for window in [4, 8]:

    shifted = (
        df.groupby(group_cols)["total_motorized"]
          .shift(1)
    )

    df[f"rolling_mean_{window}"] = (
        shifted
        .groupby(
            [df["site"], df["heading"]]
        )
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window
            ).mean()
        )
    )

    df[f"rolling_std_{window}"] = (
        shifted
        .groupby(
            [df["site"], df["heading"]]
        )
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window
            ).std()
        )
    )

# ------------------------------------------------------------
# Target:
# next 15-minute motorized demand
# ------------------------------------------------------------

df["target_next_15min"] = (
    df.groupby(group_cols)["total_motorized"]
      .shift(-1)
)

# ------------------------------------------------------------
# Vehicle composition
# ------------------------------------------------------------

composition_cols = [
    "motorcycle",
    "short_vehicle",
    "short_vehicle_towing",
    "rigid_truck_bus",
    "heavy_articulated",
]

for c in composition_cols:

    df[f"{c}_share"] = np.where(
        df["total_motorized"] > 0,
        df[c] / df["total_motorized"],
        0,
    )

# ------------------------------------------------------------
# Remove rows without sufficient history / target
# ------------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=[
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "rolling_mean_4",
        "rolling_mean_8",
        "target_next_15min",
    ]
).copy()

after = len(df)

print(f"Rows before feature filtering: {before:,}")
print(f"Rows after feature filtering:  {after:,}")
print(f"Rows removed:                  {before - after:,}")

# Save
df.to_csv(
    FEATURE_OUT,
    index=False
)

print()
print(f"Features saved to: {FEATURE_OUT}")

print()
print("FEATURE COLUMNS")
print("-" * 80)

for c in df.columns:
    print(c)

print()
print("=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)
