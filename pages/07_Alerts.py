
import streamlit as st
import pandas as pd

from state import get_all_battery_summaries


# =========================================================
# 1. PAGE HEADER
# =========================================================

st.title("Battery Alerts")
st.caption("Live battery condition, fault warnings and diagnostic alerts.")


# =========================================================
# 2. LOAD BATTERY SUMMARIES
# =========================================================

summaries = get_all_battery_summaries()

if not summaries:
    st.error("No battery data available.")
    st.stop()


# =========================================================
# 3. COUNT BATTERY CONDITIONS
# =========================================================

critical = 0
warning = 0
anomaly = 0
healthy = 0

for data in summaries.values():

    status = data["status"]

    if status == "CRITICAL":
        critical += 1

    elif status == "WARNING":
        warning += 1

    elif status == "ANOMALY":
        anomaly += 1

    else:
        healthy += 1


col1, col2, col3, col4 = st.columns(4)

col1.metric("Critical", critical)
col2.metric("Warning", warning)
col3.metric("Anomaly", anomaly)
col4.metric("Healthy", healthy)


# =========================================================
# 4. ACTIVE ALERTS
# =========================================================

st.divider()
st.subheader("Active Alerts")

active_alerts = []

for cell, data in summaries.items():

    if data["status"] in ["CRITICAL", "WARNING", "ANOMALY"]:
        active_alerts.append((cell, data))


if not active_alerts:

    st.success("No active battery alerts. All cells are healthy.")

else:

    for cell, data in active_alerts:

        saved = data.get("saved_reading")

        message = (
            f"Capacity: {data['capacity']:.3f} Ah\n\n"
            f"SOH: {data['soh']:.1f}%\n\n"
            f"Discharge Cycle: {data['cycle']}\n\n"
            f"{data['alert']}"
        )

        if data["status"] == "CRITICAL":
            st.error(f"CRITICAL | {cell}\n\n{message}")

        elif data["status"] == "WARNING":
            st.warning(f"WARNING | {cell}\n\n{message}")

        elif data["status"] == "ANOMALY":
            st.warning(f"ANOMALY | {cell}\n\n{message}")

        # Show diagnostic flags from saved measurements
        if saved:

            flags = saved.get("flags", [])

            if flags:

                with st.expander(f"Diagnostic details — {cell}"):

                    for flag in flags:
                        st.write(f"• {flag}")


# =========================================================
# 5. LATEST SAVED MEASUREMENTS
# =========================================================

st.divider()
st.subheader("Latest Saved Measurements")

saved_rows = []

for cell, data in summaries.items():

    saved = data.get("saved_reading")

    if saved:

        saved_rows.append({
            "Battery": cell,
            "Mode": saved.get("mode", "N/A"),
            "Capacity (Ah)": round(float(saved["capacity"]), 3),
            "Cycle": round(float(saved["cycle"]), 1),
            "Verdict": saved.get("verdict", "N/A"),
            "Timestamp": saved.get("timestamp", "N/A")
        })

if saved_rows:

    st.dataframe(
        pd.DataFrame(saved_rows),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No live or replay measurements have been saved yet.")


# =========================================================
# 6. ALL BATTERY CONDITIONS
# =========================================================

st.divider()
st.subheader("All Battery Conditions")

overview = []

for cell, data in summaries.items():

    overview.append({
        "Battery": cell,
        "Discharge Cycle": data["cycle"],
        "Capacity (Ah)": round(data["capacity"], 3),
        "SOH (%)": round(data["soh"], 1),
        "Condition": data["status"]
    })


st.dataframe(
    pd.DataFrame(overview),
    use_container_width=True,
    hide_index=True
)
