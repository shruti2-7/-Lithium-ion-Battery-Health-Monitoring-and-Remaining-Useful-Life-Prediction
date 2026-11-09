from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "discharge_capacity.csv"

EOL_CAPACITY = 1.5

st.markdown("""
<style>
.stApp {
    background: #080e19;
    color: #e8edf5;
}
.eda-header {
    padding: 30px;
    background: linear-gradient(120deg, #142b3d, #102c32);
    border: 1px solid #294b56;
    border-radius: 15px;
    margin-bottom: 25px;
}
.eda-label {
    color: #57d6b0;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 600;
}
.eda-header h1 {
    color: #f4f7fb;
    font-size: 38px;
}
.eda-header p {
    color: #b5c5d3;
    font-size: 15px;
}
div[data-testid="stMetric"] {
    background: #111d2c;
    border: 1px solid #293b50;
    padding: 17px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Check data/processed/discharge_capacity.csv")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="eda-header">
    <div class="eda-label">AURORA / ANALYTICS WORKFLOW / STAGE 02</div>
    <h1>Exploratory Data Analysis</h1>
    <p>
        Explore battery capacity degradation, statistical
        distributions and performance differences across
        four NASA lithium-ion battery cells.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FILTERS
# --------------------------------------------------

st.subheader("Analysis Controls")

batteries = sorted(df["battery"].unique())

selected_batteries = st.multiselect(
    "Select battery cells",
    options=batteries,
    default=batteries
)

if not selected_batteries:
    st.warning("Select at least one battery cell.")
    st.stop()

filtered_df = df[df["battery"].isin(selected_batteries)].copy()

# --------------------------------------------------
# KEY METRICS
# --------------------------------------------------

st.markdown("### Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Discharge Records", f"{len(filtered_df):,}")

with col2:
    st.metric(
        "Average Capacity",
        f"{filtered_df['capacity_Ah'].mean():.3f} Ah"
    )

with col3:
    st.metric(
        "Highest Capacity",
        f"{filtered_df['capacity_Ah'].max():.3f} Ah"
    )

with col4:
    st.metric(
        "Lowest Capacity",
        f"{filtered_df['capacity_Ah'].min():.3f} Ah"
    )

st.markdown("---")

# --------------------------------------------------
# CAPACITY DEGRADATION
# --------------------------------------------------

st.subheader("01 · Battery Capacity Degradation")

st.caption(
    "Discharge capacity across experimental cycles. "
    "The horizontal reference indicates the selected "
    "1.5 Ah End-of-Life threshold."
)

fig = go.Figure()

for battery in selected_batteries:

    battery_df = filtered_df[
        filtered_df["battery"] == battery
    ].sort_values("discharge_idx")

    fig.add_trace(go.Scatter(
        x=battery_df["discharge_idx"],
        y=battery_df["capacity_Ah"],
        mode="lines+markers",
        name=battery,
        line=dict(width=2),
        marker=dict(size=4)
    ))

fig.add_hline(
    y=EOL_CAPACITY,
    line_dash="dash",
    line_color="#ff6b6b",
    annotation_text="EOL Threshold: 1.5 Ah",
    annotation_position="top left"
)

fig.update_layout(
    template="plotly_dark",
    height=480,
    xaxis_title="Discharge Cycle Number",
    yaxis_title="Discharge Capacity (Ah)",
    hovermode="x unified",
    legend_title="Battery Cell",
    margin=dict(l=20, r=20, t=35, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# CAPACITY DISTRIBUTION
# --------------------------------------------------

st.subheader("02 · Capacity Distribution")

col1, col2 = st.columns(2)

with col1:

    fig_box = px.box(
        filtered_df,
        x="battery",
        y="capacity_Ah",
        color="battery",
        template="plotly_dark",
        title="Capacity Distribution by Battery"
    )

    fig_box.add_hline(
        y=EOL_CAPACITY,
        line_dash="dash",
        line_color="#ff6b6b"
    )

    fig_box.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Battery Cell",
        yaxis_title="Capacity (Ah)"
    )

    st.plotly_chart(fig_box, use_container_width=True)

with col2:

    fig_hist = px.histogram(
        filtered_df,
        x="capacity_Ah",
        color="battery",
        barmode="overlay",
        opacity=0.65,
        template="plotly_dark",
        title="Capacity Frequency Distribution"
    )

    fig_hist.add_vline(
        x=EOL_CAPACITY,
        line_dash="dash",
        line_color="#ff6b6b"
    )

    fig_hist.update_layout(
        height=400,
        xaxis_title="Capacity (Ah)",
        yaxis_title="Frequency"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# STATISTICAL SUMMARY
# --------------------------------------------------

st.subheader("03 · Statistical Summary")

summary = (
    filtered_df.groupby("battery")["capacity_Ah"]
    .agg(
        Records="count",
        Mean="mean",
        Standard_Deviation="std",
        Minimum="min",
        Median="median",
        Maximum="max"
    )
    .round(4)
)

st.dataframe(
    summary,
    use_container_width=True
)

# --------------------------------------------------
# TEMPERATURE ANALYSIS
# --------------------------------------------------

st.markdown("---")

st.subheader("04 · Ambient Temperature Analysis")

fig_temp = px.scatter(
    filtered_df,
    x="ambient_temperature_C",
    y="capacity_Ah",
    color="battery",
    template="plotly_dark",
    title="Ambient Temperature vs Discharge Capacity",
    hover_data=["discharge_idx"]
)

fig_temp.update_layout(
    height=420,
    xaxis_title="Ambient Temperature (°C)",
    yaxis_title="Discharge Capacity (Ah)"
)

st.plotly_chart(fig_temp, use_container_width=True)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

st.markdown("---")

with st.expander("View Extracted Dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Filtered Dataset",
        data=filtered_df.to_csv(index=False),
        file_name="aurora_filtered_battery_data.csv",
        mime="text/csv"
    )

st.caption(
    "SOURCE: NASA Prognostics Center of Excellence (PCoE) "
    "| Experimental battery data | AURORA"
)