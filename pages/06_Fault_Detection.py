
import streamlit as st
import pandas as pd
import numpy as np

from state import (
    BATTERY_CELLS,
    RATED_CAPACITY,
    WARNING_THRESHOLD,
    EOL_THRESHOLD,
    IR_WARNING_THRESHOLD,
    load_cycles,
    save_reading,
    get_previous_reading,
    get_all_battery_summaries
)


# =========================================================
# 1. PAGE HEADER
# =========================================================

st.title("Fault Detection")
st.caption("Battery condition assessment and measurement analysis.")

mode = st.radio(
    "Select Analysis Mode",
    ["Dataset Replay", "Live Measurement"],
    horizontal=True
)


# =========================================================
# 2. SHARED HELPER FUNCTIONS
# =========================================================

def get_capacity_status(capacity):

    if capacity <= EOL_THRESHOLD:
        return "CRITICAL"

    elif capacity <= WARNING_THRESHOLD:
        return "WARNING"

    return "OK"


def estimate_battery_life(battery_df, measured_capacity):

    battery_df = battery_df.sort_values("cycle")

    x = battery_df["cycle"].to_numpy(dtype=float)
    y = battery_df["capacity"].to_numpy(dtype=float)

    if len(x) < 2:
        return None, None

    slope, intercept = np.polyfit(x, y, 1)

    if abs(slope) < 1e-10 or slope >= 0:
        return None, None

    effective_cycle = (measured_capacity - intercept) / slope

    effective_cycle = float(
        np.clip(effective_cycle, x.min(), x.max())
    )

    eol_cycle = (EOL_THRESHOLD - intercept) / slope

    rul = max(0, eol_cycle - effective_cycle)

    return effective_cycle, rul


def show_verdict(verdict):

    if verdict == "CRITICAL":
        st.error("CRITICAL — Battery has reached its end-of-life threshold.")

    elif verdict == "WARNING":
        st.warning("WARNING — Battery capacity is below the normal range.")

    elif verdict == "ANOMALY":
        st.error("ANOMALY — Unusual battery behaviour detected.")

    else:
        st.success("OK — Battery is operating within the normal range.")


# =========================================================
# 3. LOAD DATA
# =========================================================

df = load_cycles()

if df.empty:
    st.error("No discharge-cycle data available.")
    st.stop()


# =========================================================
# 4. DATASET REPLAY MODE
# =========================================================

if mode == "Dataset Replay":

    st.subheader("Recorded Dataset Inspection")

    st.caption(
        "Inspect historical discharge-cycle measurements from the NASA dataset."
    )

    selected_cell = st.selectbox(
        "Select Battery Cell",
        BATTERY_CELLS,
        key="replay_cell"
    )

    battery_df = df[df["cell"] == selected_cell].copy()
    battery_df = battery_df.sort_values("cycle")

    if battery_df.empty:
        st.error("No data available for this cell.")
        st.stop()

    min_cycle = int(battery_df["cycle"].min())
    max_cycle = int(battery_df["cycle"].max())

    selected_cycle = st.slider(
        "Select Discharge Cycle",
        min_value=min_cycle,
        max_value=max_cycle,
        value=max_cycle,
        key="replay_cycle"
    )

    selected_row = battery_df[
        battery_df["cycle"] == selected_cycle
    ]

    if selected_row.empty:
        st.error("Selected cycle is unavailable.")
        st.stop()

    measured_capacity = float(
        selected_row.iloc[0]["capacity"]
    )

    soh = (measured_capacity / RATED_CAPACITY) * 100

    verdict = get_capacity_status(measured_capacity)

    effective_cycle, rul = estimate_battery_life(
        battery_df,
        measured_capacity
    )

    st.divider()

    st.subheader("Selected Battery Condition")

    col1, col2, col3 = st.columns(3)

    col1.metric("Discharge Cycle", selected_cycle)
    col2.metric("Capacity", f"{measured_capacity:.3f} Ah")
    col3.metric("State of Health", f"{soh:.1f}%")

    if rul is not None:
        col1, col2 = st.columns(2)

        col1.metric("Estimated Remaining Life", f"{rul:.0f} cycles")
        col2.metric("Estimated EOL Cycle", f"{effective_cycle + rul:.0f}" if effective_cycle is not None else "N/A")

    show_verdict(verdict)

    st.subheader("Capacity Degradation")

    st.line_chart(
        battery_df.set_index("cycle")["capacity"],
        y_label="Capacity (Ah)",
        x_label="Discharge Cycle"
    )

    st.divider()

    st.subheader("Save Selected Reading")

    st.caption(
        "Save this historical measurement so it can be reflected in Battery Analysis and Alerts."
    )

    if st.button("Save Dataset Reading", key="save_replay"):

        save_reading(
            cell=selected_cell,
            capacity=measured_capacity,
            cycle=selected_cycle,
            verdict=verdict,
            mode="Dataset replay"
        )

        st.success("Dataset reading saved successfully.")
        st.rerun()


# =========================================================
# 5. LIVE MEASUREMENT MODE
# =========================================================

elif mode == "Live Measurement":

    st.subheader("Live Battery Measurement")

    st.caption(
        "Enter a battery measurement to assess its present condition."
    )

    with st.form("live_measurement_form"):

        selected_cell = st.selectbox(
            "Select Battery Cell",
            BATTERY_CELLS,
            key="live_cell"
        )

        measured_capacity = st.number_input(
            "Measured Capacity (Ah)",
            min_value=0.01,
            max_value=10.0,
            value=1.90,
            step=0.01,
            help="end-of-charge mAh from the charger divided by 1000, e.g. 4100 mAh -> 4.10 Ah"
        )

        internal_resistance = st.number_input(
            "Internal Resistance (mOhm) — Optional",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        voltage_delta = st.number_input(
            "Cell Voltage Delta (mV) — Optional",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        submitted = st.form_submit_button(
            "Analyse Measurement",
            use_container_width=True
        )

    if submitted:

        battery_df = df[df["cell"] == selected_cell].copy()
        battery_df = battery_df.sort_values("cycle")

        if battery_df.empty:
            st.error("No historical data available for this cell.")

        else:

            # ---------------------------------------------
            # SOH CALCULATION
            # ---------------------------------------------

            soh = (measured_capacity / RATED_CAPACITY) * 100

            # ---------------------------------------------
            # EFFECTIVE CYCLE AND RUL
            # ---------------------------------------------

            effective_cycle, rul = estimate_battery_life(
                battery_df,
                measured_capacity
            )

            if effective_cycle is None:
                effective_cycle = float(
                    battery_df["cycle"].max()
                )

            # ---------------------------------------------
            # CAPACITY CONDITION
            # ---------------------------------------------

            verdict = get_capacity_status(measured_capacity)

            flags = []

            # ---------------------------------------------
            # ANOMALY DETECTION: Z-SCORE
            # ---------------------------------------------

            recorded_capacities = battery_df["capacity"].to_numpy(
                dtype=float
            )

            mean_capacity = np.mean(recorded_capacities)
            std_capacity = np.std(recorded_capacities)

            if std_capacity > 0:

                z_score = (
                    measured_capacity - mean_capacity
                ) / std_capacity

                if abs(z_score) > 3:
                    flags.append(
                        f"Unusual capacity reading (Z-score: {z_score:.2f})"
                    )

            # ---------------------------------------------
            # ANOMALY DETECTION: DROP FROM PREVIOUS READING
            # ---------------------------------------------

            previous = get_previous_reading(selected_cell)

            if previous is not None:

                previous_capacity = float(previous["capacity"])

                if previous_capacity > 0:

                    percentage_drop = (
                        (previous_capacity - measured_capacity)
                        / previous_capacity
                    ) * 100

                    if percentage_drop > 3:
                        flags.append(
                            f"Capacity dropped by {percentage_drop:.1f}% compared with the previous saved reading"
                        )

            # ---------------------------------------------
            # INTERNAL RESISTANCE CHECK
            # ---------------------------------------------

            if (
                internal_resistance > 0
                and internal_resistance > IR_WARNING_THRESHOLD
            ):

                flags.append(
                    f"High internal resistance: {internal_resistance:.1f} mOhm"
                )

            # ---------------------------------------------
            # VOLTAGE DELTA CHECK
            # ---------------------------------------------

            if voltage_delta > 0:

                flags.append(
                    f"Recorded cell voltage delta: {voltage_delta:.1f} mV"
                )

            # ---------------------------------------------
            # FINAL VERDICT
            # ---------------------------------------------

            if verdict == "OK" and any(
                "Unusual capacity" in flag
                or "Capacity dropped" in flag
                for flag in flags
            ):
                verdict = "ANOMALY"

            # ---------------------------------------------
            # SAVE MEASUREMENT
            # ---------------------------------------------

            save_reading(
                cell=selected_cell,
                capacity=measured_capacity,
                cycle=effective_cycle,
                verdict=verdict,
                internal_resistance=(
                    internal_resistance
                    if internal_resistance > 0 else None
                ),
                voltage_delta=(
                    voltage_delta
                    if voltage_delta > 0 else None
                ),
                mode="Live measurement",
                flags=flags
            )

            # ---------------------------------------------
            # DISPLAY RESULTS
            # ---------------------------------------------

            st.session_state["live_result"] = {
                "cell": selected_cell,
                "capacity": measured_capacity,
                "soh": soh,
                "cycle": effective_cycle,
                "rul": rul,
                "verdict": verdict,
                "flags": flags
            }

            st.rerun()


# =========================================================
# 6. DISPLAY SAVED LIVE RESULTS
# =========================================================

if mode == "Live Measurement":

    result = st.session_state.get("live_result")

    if result is not None:

        st.divider()

        st.subheader("Measurement Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "State of Health",
            f"{result['soh']:.1f}%"
        )

        col2.metric(
            "Effective Cycle",
            f"{result['cycle']:.1f}"
        )

        col3.metric(
            "Remaining Useful Life",
            f"{result['rul']:.0f} cycles"
            if result["rul"] is not None
            else "Not estimable"
        )

        show_verdict(result["verdict"])

        if result["flags"]:

            st.subheader("Diagnostic Flags")

            for flag in result["flags"]:
                st.warning(flag)

        st.success(
            "Measurement saved. It is now available to the Battery Analysis and Alerts pages."
        )


# =========================================================
# 7. BATTERY CONDITION OVERVIEW
# =========================================================

st.divider()

st.subheader("Battery Condition Overview")

summaries = get_all_battery_summaries()

overview = []

for cell, data in summaries.items():

    overview.append({
        "Battery": cell,
        "Latest Cycle": data["cycle"],
        "Capacity (Ah)": round(data["capacity"], 3),
        "SOH (%)": round(data["soh"], 1),
        "Condition": data["status"]
    })

st.dataframe(
    pd.DataFrame(overview),
    use_container_width=True,
    hide_index=True
)
