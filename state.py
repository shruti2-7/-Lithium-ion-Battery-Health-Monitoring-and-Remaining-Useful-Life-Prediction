from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import streamlit as st


# =========================================================
# AURORA BATTERY CONFIGURATION
# =========================================================

BATTERY_CELLS = ["B0005", "B0006", "B0007", "B0018"]

RATED_CAPACITY = 2.0
WARNING_THRESHOLD = 1.8
EOL_THRESHOLD = 1.5
IR_WARNING_THRESHOLD = 50.0

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "processed" / "discharge_capacity.csv"
READINGS_FILE = BASE_DIR / "readings.json"


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_cycles():

    if not DATA_FILE.exists():
        return pd.DataFrame(
            columns=["cell", "cycle", "capacity", "ambient_temp"]
        )

    df = pd.read_csv(DATA_FILE)

    required = [
        "battery",
        "discharge_idx",
        "capacity_Ah"
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing dataset column: {column}")

    df = df.rename(columns={
        "battery": "cell",
        "discharge_idx": "cycle",
        "capacity_Ah": "capacity",
        "ambient_temperature_C": "ambient_temp"
    })

    if "ambient_temp" not in df.columns:
        df["ambient_temp"] = None

    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    df = df.dropna(subset=["cell", "cycle", "capacity"])

    df["cell"] = df["cell"].astype(str)

    df = df.sort_values(["cell", "cycle"])

    return df.reset_index(drop=True)


# =========================================================
# SAVED READINGS
# =========================================================

def load_readings():

    if not READINGS_FILE.exists():
        return []

    try:
        with open(READINGS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        return []


def save_reading(
    cell,
    capacity,
    cycle,
    verdict,
    internal_resistance=None,
    voltage_delta=None,
    mode="Dataset replay",
    flags=None
):

    readings = load_readings()

    reading = {
        "cell": str(cell),
        "capacity": float(capacity),
        "cycle": int(cycle) if cycle is not None else None,
        "verdict": str(verdict).upper(),
        "internal_resistance": (
            float(internal_resistance)
            if internal_resistance is not None
            else None
        ),
        "voltage_delta": (
            float(voltage_delta)
            if voltage_delta is not None
            else None
        ),
        "mode": mode,
        "flags": flags if flags is not None else [],
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }

    readings.append(reading)

    with open(READINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(readings, file, indent=4)

    return reading


def get_latest_reading(cell):

    readings = load_readings()

    cell_readings = [
        reading for reading in readings
        if reading.get("cell") == cell
    ]

    if not cell_readings:
        return None

    return cell_readings[-1]


def get_previous_reading(cell):

    readings = load_readings()

    cell_readings = [
        reading for reading in readings
        if reading.get("cell") == cell
    ]

    if not cell_readings:
        return None

    return cell_readings[-1]


# =========================================================
# BATTERY CLASSIFICATION
# =========================================================

def classify_capacity(capacity):

    if capacity is None:
        return "Unknown"

    capacity = float(capacity)

    if capacity <= EOL_THRESHOLD:
        return "Critical"

    elif capacity <= WARNING_THRESHOLD:
        return "Moderate"

    return "Good"


def analyze_battery(cell):

    latest = get_latest_reading(cell)

    if latest:
        capacity = latest.get("capacity")
        cycle = latest.get("cycle")

        return {
            "cell": cell,
            "capacity": capacity,
            "cycle": cycle,
            "status": classify_capacity(capacity),
            "source": latest.get("mode", "Saved reading"),
            "reading": latest
        }

    df = load_cycles()

    cell_df = df[df["cell"] == cell]

    if cell_df.empty:
        return None

    latest_row = cell_df.iloc[-1]

    capacity = float(latest_row["capacity"])
    cycle = int(latest_row["cycle"])

    return {
        "cell": cell,
        "capacity": capacity,
        "cycle": cycle,
        "status": classify_capacity(capacity),
        "source": "NASA dataset",
        "reading": None
    }


# =========================================================
# ALL BATTERY SUMMARIES
# =========================================================

def get_all_battery_summaries():

    summaries = []

    for cell in BATTERY_CELLS:

        result = analyze_battery(cell)

        if result is not None:
            summaries.append(result)

    return summaries