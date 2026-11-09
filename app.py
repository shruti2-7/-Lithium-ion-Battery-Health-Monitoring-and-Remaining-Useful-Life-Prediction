
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from state import (
    BATTERY_CELLS,
    RATED_CAPACITY,
    WARNING_THRESHOLD,
    EOL_THRESHOLD,
    IR_WARNING_THRESHOLD,
    save_reading,
    get_latest_reading,
    get_previous_reading
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AURORA | Battery Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

DATA_PATH = Path("data/processed/discharge_capacity.csv")

BATTERIES = BATTERY_CELLS
WARNING_CAPACITY = WARNING_THRESHOLD
EOL_CAPACITY = EOL_THRESHOLD


# =========================================================
# DATA FUNCTIONS
# =========================================================

@st.cache_data
def load_battery_data():

    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "battery",
        "discharge_idx",
        "capacity_Ah",
        "ambient_temperature_C"
    ]

    if not all(column in df.columns for column in required_columns):
        return None

    df = df.dropna(
        subset=["battery", "discharge_idx", "capacity_Ah"]
    ).copy()

    df["battery"] = df["battery"].astype(str)

    df["discharge_idx"] = pd.to_numeric(
        df["discharge_idx"], errors="coerce"
    )

    df["capacity_Ah"] = pd.to_numeric(
        df["capacity_Ah"], errors="coerce"
    )

    df["ambient_temperature_C"] = pd.to_numeric(
        df["ambient_temperature_C"], errors="coerce"
    )

    df = df.dropna(
        subset=["discharge_idx", "capacity_Ah"]
    ).copy()

    df["discharge_idx"] = df["discharge_idx"].astype(int)

    return df.sort_values(
        ["battery", "discharge_idx"]
    ).reset_index(drop=True)


def classify_battery(capacity):

    if capacity is None:
        return "Unknown"

    if capacity <= EOL_CAPACITY:
        return "Critical"

    if capacity <= WARNING_CAPACITY:
        return "Moderate"

    return "Good"


def get_status_html(status):

    status_classes = {
        "Good": "status-good",
        "Moderate": "status-moderate",
        "Critical": "status-critical",
        "Anomaly": "status-anomaly"
    }

    css_class = status_classes.get(status, "status-good")

    return (
        f'<span class="status-pill {css_class}">'
        f'{status.upper()}</span>'
    )


# =========================================================
# SHARED CURRENT MEASUREMENT
# =========================================================

def get_cell_measurement(df, battery):

    saved = get_latest_reading(battery)

    if saved is not None:

        capacity = float(saved["capacity"])

        cycle_value = saved.get("cycle")

        cycle = (
            int(round(float(cycle_value)))
            if cycle_value is not None
            else None
        )

        verdict = str(
            saved.get("verdict", "HEALTHY")
        ).upper()

        if verdict == "CRITICAL":
            status = "Critical"
        elif verdict in ["WARNING", "MODERATE"]:
            status = "Moderate"
        elif verdict == "ANOMALY":
            status = "Anomaly"
        else:
            status = "Good"

        return {
            "capacity": capacity,
            "cycle": cycle,
            "status": status,
            "source": saved.get("mode", "Saved measurement"),
            "saved": True
        }

    battery_df = df[
        df["battery"] == battery
    ].sort_values("discharge_idx")

    if battery_df.empty:
        return None

    latest_row = battery_df.iloc[-1]

    capacity = float(latest_row["capacity_Ah"])
    cycle = int(latest_row["discharge_idx"])

    return {
        "capacity": capacity,
        "cycle": cycle,
        "status": classify_battery(capacity),
        "source": "NASA Dataset · Latest experimental cycle",
        "saved": False
    }


# =========================================================
# ALERT SUMMARY
# =========================================================

def get_individual_cycle_summary(df):

    rows = []

    for battery in BATTERIES:

        measurement = get_cell_measurement(df, battery)

        if measurement is None:
            continue

        capacity = measurement["capacity"]
        cycle = measurement["cycle"]

        soh = capacity / RATED_CAPACITY * 100

        rows.append({
            "Battery": battery,
            "Selected Cycle": cycle,
            "Capacity (Ah)": capacity,
            "SOH (%)": soh,
            "Status": measurement["status"],
            "Source": measurement["source"]
        })

    return pd.DataFrame(rows)


def get_alert_level(summary):

    if summary is None or summary.empty:
        return "none"

    statuses = set(summary["Status"].tolist())

    if "Critical" in statuses:
        return "critical"

    if "Moderate" in statuses:
        return "moderate"

    if "Anomaly" in statuses:
        return "anomaly"

    return "good"


# =========================================================
# GLOBAL THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            ellipse at 85% 0%,
            rgba(35, 112, 112, 0.10),
            transparent 40%
        ),
        linear-gradient(
            160deg,
            #07121D 0%,
            #0A1624 60%,
            #0E1828 100%
        );
    color: #E6EEF6;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3 {
    color: #F1F7FD;
    letter-spacing: -0.3px;
}

h1 {
    font-weight: 700;
    letter-spacing: -0.8px;
    margin-bottom: 0.2rem;
}

h2, h3 {
    font-weight: 650;
}

hr {
    border-color: #1F3449;
    margin: 1.6rem 0;
}


/* =====================================================
   MAIN HOMEPAGE AURORA TITLE
   METALLIC PURPLE - PINK - BLUE GRADIENT
   ===================================================== */

.home-title {
    font-size: clamp(3.5rem, 7vw, 5rem);
    font-weight: 900;
    letter-spacing: 0.18em;
    line-height: 1.1;
    margin-bottom: 16px;

    background: linear-gradient(
        105deg,
        #B69AFF 0%,
        #E6A4D7 35%,
        #91BDF5 68%,
        #B9A5F5 100%
    );

    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;

    filter: drop-shadow(
        0 0 7px rgba(174, 137, 255, 0.22)
    );
}


/* =====================================================
   SIDEBAR
   ===================================================== */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #08131F 0%,
        #0C1828 60%,
        #0E1A2C 100%
    );
    border-right: 1px solid rgba(92, 135, 162, 0.18);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.6rem;
}

.sidebar-brand {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 4px 0 6px 0;
}

.sidebar-brand-icon {
    width: 40px;
    height: 40px;
    margin-bottom: 10px;
}

.sidebar-brand-name {
    color: #F1F7FD;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 9px;
    padding-left: 9px;
}

.sidebar-brand-subtitle {
    color: #7F9BB4;
    font-size: 9.5px;
    letter-spacing: 2.2px;
    font-weight: 600;
    margin-top: 8px;
}

.sidebar-brand-line {
    width: 100%;
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(32, 227, 207, 0.6),
        rgba(155, 123, 255, 0.6),
        transparent
    );
    margin: 20px 0 18px 0;
}

.nav-section-label {
    color: #8FB3CC;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 4px 12px 8px 12px;
    user-select: none;
    cursor: default;
    text-shadow: 0 0 10px rgba(32, 227, 207, 0.35);
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
    border-radius: 9px;
    padding: 9px 12px;
    margin: 3px 0;
    color: #D3DFEA !important;
    background: transparent;
    border: 1px solid transparent;
    transition: background 0.2s ease, border 0.2s ease;
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
    background: linear-gradient(
        100deg,
        rgba(32, 227, 207, 0.14),
        rgba(155, 123, 255, 0.12)
    ) !important;
    border: 1px solid rgba(32, 227, 207, 0.60);
    box-shadow:
        0 0 10px rgba(32, 227, 207, 0.25),
        0 0 22px rgba(155, 123, 255, 0.18);
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:active,
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:focus,
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:focus-visible {
    background: linear-gradient(
        100deg,
        rgba(32, 227, 207, 0.22),
        rgba(155, 123, 255, 0.20)
    ) !important;
    border: 1px solid #20E3CF;
    box-shadow:
        0 0 14px rgba(32, 227, 207, 0.45),
        0 0 30px rgba(155, 123, 255, 0.30);
    color: #FFFFFF !important;
    outline: none;
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: linear-gradient(
        100deg,
        rgba(32, 227, 207, 0.20),
        rgba(155, 123, 255, 0.18)
    ) !important;
    border: 1px solid #20E3CF;
    box-shadow:
        0 0 12px rgba(32, 227, 207, 0.45),
        0 0 26px rgba(155, 123, 255, 0.28),
        inset 0 0 10px rgba(32, 227, 207, 0.12);
    color: #FFFFFF !important;
    animation: navGlowPulse 2.4s ease-in-out infinite;
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] svg {
    color: #20E3CF;
    filter: drop-shadow(0 0 6px rgba(32, 227, 207, 0.9));
}

@keyframes navGlowPulse {
    0%, 100% {
        box-shadow:
            0 0 12px rgba(32, 227, 207, 0.40),
            0 0 26px rgba(155, 123, 255, 0.25);
    }
    50% {
        box-shadow:
            0 0 18px rgba(32, 227, 207, 0.65),
            0 0 36px rgba(155, 123, 255, 0.40);
    }
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] svg {
    color: #9CC3D9;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: 1px solid #1F3449;
    border-radius: 10px;
    background: rgba(14, 26, 42, 0.6);
}


/* =====================================================
   WELCOME PAGE - GLOWING STEP CARDS
   ===================================================== */

.welcome-step {
    position: relative;
    background: linear-gradient(
        145deg,
        rgba(20, 30, 53, 0.98),
        rgba(12, 22, 39, 0.98)
    );
    border-radius: 16px;
    padding: 28px 25px;
    min-height: 205px;
    transition:
        transform 0.3s ease,
        box-shadow 0.3s ease,
        border-color 0.3s ease;
    overflow: visible;
}

/* PURPLE - MONITOR */

.welcome-monitor {
    border: 1px solid rgba(168, 113, 255, 0.65);
    box-shadow:
        0 0 12px rgba(168, 113, 255, 0.18),
        0 0 28px rgba(168, 113, 255, 0.12),
        inset 0 0 20px rgba(168, 113, 255, 0.06);
}

.welcome-monitor .welcome-step-number {
    color: #C6A1FF;
}

.welcome-monitor .welcome-step-title {
    color: #D6B8FF;
    text-shadow: 0 0 14px rgba(168, 113, 255, 0.55);
}

.welcome-monitor:hover {
    transform: translateY(-9px) scale(1.025);
    border-color: #B98AFF;
    box-shadow:
        0 0 18px rgba(168, 113, 255, 0.55),
        0 0 38px rgba(168, 113, 255, 0.35),
        0 14px 35px rgba(0, 0, 0, 0.35),
        inset 0 0 24px rgba(168, 113, 255, 0.12);
}


/* PINK - ANALYSE */

.welcome-analyse {
    border: 1px solid rgba(255, 105, 180, 0.65);
    box-shadow:
        0 0 12px rgba(255, 105, 180, 0.18),
        0 0 28px rgba(255, 105, 180, 0.12),
        inset 0 0 20px rgba(255, 105, 180, 0.06);
}

.welcome-analyse .welcome-step-number {
    color: #FF9ACD;
}

.welcome-analyse .welcome-step-title {
    color: #FFB1D8;
    text-shadow: 0 0 14px rgba(255, 105, 180, 0.55);
}

.welcome-analyse:hover {
    transform: translateY(-9px) scale(1.025);
    border-color: #FF85C4;
    box-shadow:
        0 0 18px rgba(255, 105, 180, 0.55),
        0 0 38px rgba(255, 105, 180, 0.35),
        0 14px 35px rgba(0, 0, 0, 0.35),
        inset 0 0 24px rgba(255, 105, 180, 0.12);
}


/* BLUE - PREDICT */

.welcome-predict {
    border: 1px solid rgba(69, 183, 255, 0.65);
    box-shadow:
        0 0 12px rgba(69, 183, 255, 0.18),
        0 0 28px rgba(69, 183, 255, 0.12),
        inset 0 0 20px rgba(69, 183, 255, 0.06);
}

.welcome-predict .welcome-step-number {
    color: #82D1FF;
}

.welcome-predict .welcome-step-title {
    color: #A6DEFF;
    text-shadow: 0 0 14px rgba(69, 183, 255, 0.55);
}

.welcome-predict:hover {
    transform: translateY(-9px) scale(1.025);
    border-color: #65C5FF;
    box-shadow:
        0 0 18px rgba(69, 183, 255, 0.55),
        0 0 38px rgba(69, 183, 255, 0.35),
        0 14px 35px rgba(0, 0, 0, 0.35),
        inset 0 0 24px rgba(69, 183, 255, 0.12);
}


/* SHARED CARD TEXT */

.welcome-step-number {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 18px;
}

.welcome-step-title {
    font-size: 23px;
    font-weight: 750;
    letter-spacing: 1.5px;
    margin-bottom: 15px;
}

.welcome-step-description {
    color: #B6C6D7;
    font-size: 14px;
    line-height: 1.8;
}


/* PROJECT DESCRIPTION */

.welcome-description {
    color: #B6C6D7;
    font-size: 15px;
    line-height: 1.9;
    max-width: 1050px;
}

.welcome-project-label {
    display: block;
    width: 100%;
    height: auto;
    overflow: visible;
    line-height: 1.6;
    padding: 4px 0 8px 0;
    margin: 0;
    color: #8FA9C0;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
}


/* =====================================================
   STATUS PILLS
   ===================================================== */

.status-pill {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    border: 1px solid;
}

.status-good {
    color: #55E6B1;
    background: rgba(38, 190, 133, 0.12);
    border-color: rgba(38, 190, 133, 0.50);
}

.status-moderate {
    color: #FFC66D;
    background: rgba(255, 177, 66, 0.12);
    border-color: rgba(255, 177, 66, 0.50);
}

.status-critical {
    color: #FF7D89;
    background: rgba(255, 77, 97, 0.13);
    border-color: rgba(255, 77, 97, 0.55);
}

.status-anomaly {
    color: #C6A6FF;
    background: rgba(155, 123, 255, 0.13);
    border-color: rgba(155, 123, 255, 0.55);
}


/* =====================================================
   CARDS
   ===================================================== */

.info-card {
    background: linear-gradient(145deg, #13243A, #0F1C2E);
    border: 1px solid #2A4259;
    border-radius: 14px;
    padding: 24px;
    min-height: 160px;
}

.info-card h3 {
    color: #45B7FF;
    font-size: 18px;
}

.info-card p {
    color: #AFC1D2;
    line-height: 1.7;
    font-size: 14px;
}


/* =====================================================
   METRICS
   ===================================================== */

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #13243A, #0F1C2E);
    border: 1px solid #2A4259;
    padding: 16px 18px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #9DB3C7;
}

div[data-testid="stMetricValue"] {
    color: #20E3CF;
}


/* =====================================================
   BUTTONS
   ===================================================== */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #087F88, #1676B5);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 1px solid #20E3CF;
    color: white;
}


/* =====================================================
   DATAFRAMES & EXPANDERS
   ===================================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #294158;
    border-radius: 10px;
}

[data-testid="stExpander"] {
    border: 1px solid #294158;
    border-radius: 12px;
    background: rgba(15, 28, 44, 0.55);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR BRANDING
# =========================================================

with st.sidebar:

    sidebar_brand_html = (
        '<div class="sidebar-brand">'
        '<svg class="sidebar-brand-icon" viewBox="0 0 40 40" fill="none" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<rect x="9" y="6" width="22" height="30" rx="4" '
        'stroke="#20E3CF" stroke-width="2"/>'
        '<rect x="15" y="2.5" width="10" height="3.5" rx="1.2" '
        'fill="#20E3CF"/>'
        '<path d="M22 11 L15.5 22 L20 22 L18 30 L24.5 18.5 L20 18.5 Z" '
        'fill="#45B7FF"/>'
        '</svg>'
        '<div class="sidebar-brand-name">AURORA</div>'
        '<div class="sidebar-brand-subtitle">BATTERY INTELLIGENCE SYSTEM</div>'
        '<div class="sidebar-brand-line"></div>'
        '</div>'
    )

    st.markdown(
        sidebar_brand_html,
        unsafe_allow_html=True
    )

    st.caption("NASA PCoE · LITHIUM-ION BATTERY DATA")


# =========================================================
# DASHBOARD / WELCOME PAGE
# =========================================================

def dashboard():

    # CUSTOM METALLIC AURORA GRADIENT HEADING
    # ORIGINAL AURORA TITLE STYLING PRESERVED
    st.markdown(
        '<div class="home-title">AURORA</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Lithium-Ion Battery Health Monitoring "
        "and Remaining Useful Life Prediction"
    )

    st.markdown(
        """
        <div class="welcome-description">
            AURORA is a battery intelligence system designed to monitor
            lithium-ion battery health, analyse capacity degradation and
            predict the Remaining Useful Life (RUL) of individual battery cells.
            Using experimental data from the NASA PCoE Battery Dataset,
            the system evaluates battery performance, identifies abnormal
            conditions and estimates how many discharge cycles remain before
            a battery reaches its end-of-life threshold.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### How AURORA Works")

    st.markdown(
        """
        <p style="
            color: #8FA9C0;
            font-size: 13px;
            margin-top: -8px;
            margin-bottom: 25px;
        ">
            Three key stages that turn battery data into actionable insights.
        </p>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:

        st.markdown(
            """
            <div class="welcome-step welcome-monitor">
                <div class="welcome-step-number">01 / MONITORING</div>
                <div class="welcome-step-title">MONITOR</div>
                <div class="welcome-step-description">
                    Track battery capacity, State of Health (SOH)
                    and operating condition across individual cells.
                    Identify healthy, warning and critical conditions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="welcome-step welcome-analyse">
                <div class="welcome-step-number">02 / ANALYSIS</div>
                <div class="welcome-step-title">ANALYSE</div>
                <div class="welcome-step-description">
                    Examine NASA experimental discharge-cycle data,
                    understand capacity degradation trends and
                    identify unusual battery behaviour.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="welcome-step welcome-predict">
                <div class="welcome-step-number">03 / PREDICTION</div>
                <div class="welcome-step-title">PREDICT</div>
                <div class="welcome-step-description">
                    Use predictive modelling to estimate Remaining
                    Useful Life and determine the cycles remaining
                    before the battery reaches its end-of-life threshold.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown(
        """
        <div style="
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 12px;
            color: #829BB2;
            font-size: 11px;
            letter-spacing: 1px;
            padding-top: 4px;
        ">
            <span>DATA SOURCE: NASA PCoE BATTERY DATA SET</span>
            <span>CELLS: B0005 · B0006 · B0007 · B0018</span>
            <span>END-OF-LIFE THRESHOLD: 1.5 Ah</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# BATTERY INFORMATION
# NASA DATASET ONLY — NO LIVE MEASUREMENTS
# =========================================================

def battery_information():

    st.title("Battery Information")

    st.caption(
        "Experimental reference data for the four lithium-ion cells "
        "from the NASA PCoE Battery Data Set."
    )

    df = load_battery_data()

    if df is None:
        st.error("NASA battery dataset could not be loaded.")
        return

    st.divider()

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Battery Cells", len(BATTERIES))
    c2.metric("Rated Capacity", f"{RATED_CAPACITY:.1f} Ah")
    c3.metric("End-of-Life Threshold", f"{EOL_CAPACITY:.1f} Ah")

    st.divider()

    st.subheader("Individual Cell Information")

    st.caption(
        "Select a battery cell to view its original NASA experimental "
        "discharge measurements and capacity degradation history."
    )

    for battery in BATTERIES:

        battery_df = (
            df[df["battery"] == battery]
            .sort_values("discharge_idx")
            .copy()
        )

        if battery_df.empty:
            continue

        initial_capacity = float(
            battery_df["capacity_Ah"].iloc[0]
        )

        final_capacity = float(
            battery_df["capacity_Ah"].iloc[-1]
        )

        total_cycles = int(
            battery_df["discharge_idx"].nunique()
        )

        first_cycle = int(
            battery_df["discharge_idx"].min()
        )

        last_cycle = int(
            battery_df["discharge_idx"].max()
        )

        lowest_capacity = float(
            battery_df["capacity_Ah"].min()
        )

        highest_capacity = float(
            battery_df["capacity_Ah"].max()
        )

        average_temperature = float(
            battery_df["ambient_temperature_C"].mean()
        )

        capacity_loss = initial_capacity - final_capacity

        dataset_soh = (
            final_capacity / RATED_CAPACITY
        ) * 100

        with st.expander(
            f"{battery} · {total_cycles} NASA Discharge Cycles",
            expanded=False
        ):

            # Critical/Warning/Good status display intentionally removed.
            # This page presents historical NASA reference measurements only.

            st.markdown("#### Experimental Capacity Information")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Initial Capacity",
                f"{initial_capacity:.3f} Ah"
            )

            m2.metric(
                "Final NASA Capacity",
                f"{final_capacity:.3f} Ah"
            )

            m3.metric(
                "Dataset SOH",
                f"{dataset_soh:.1f}%"
            )

            m4.metric(
                "Capacity Loss",
                f"{capacity_loss:.3f} Ah"
            )

            st.markdown("#### NASA Experimental Cycle Information")

            d1, d2, d3, d4 = st.columns(4)

            d1.metric("Discharge Cycles", total_cycles)
            d2.metric("First Cycle", first_cycle)
            d3.metric("Last Cycle", last_cycle)

            d4.metric(
                "Average Temperature",
                f"{average_temperature:.1f} °C"
            )

            d1, d2 = st.columns(2)

            d1.metric(
                "Highest Capacity",
                f"{highest_capacity:.3f} Ah"
            )

            d2.metric(
                "Lowest Capacity",
                f"{lowest_capacity:.3f} Ah"
            )

            st.markdown("#### NASA Capacity Degradation History")

            fig = px.line(
                battery_df,
                x="discharge_idx",
                y="capacity_Ah",
                markers=True,
                labels={
                    "discharge_idx": "Discharge Cycle",
                    "capacity_Ah": "Capacity (Ah)"
                },
                title=f"{battery} · NASA Experimental Capacity"
            )

            fig.add_hline(
                y=RATED_CAPACITY,
                line_dash="dash",
                line_color="#20E3CF",
                annotation_text="Rated Capacity (2.0 Ah)"
            )

            fig.add_hline(
                y=WARNING_CAPACITY,
                line_dash="dash",
                line_color="orange",
                annotation_text="Warning Threshold (1.8 Ah)"
            )

            fig.add_hline(
                y=EOL_CAPACITY,
                line_dash="dash",
                line_color="red",
                annotation_text="End-of-Life Threshold (1.5 Ah)"
            )

            fig.update_traces(
                line=dict(width=2.5),
                marker=dict(size=5),
                hovertemplate=(
                    "Discharge Cycle: %{x}<br>"
                    "Capacity: %{y:.3f} Ah"
                    "<extra></extra>"
                )
            )

            fig.update_layout(
                template="plotly_dark",
                height=380,
                margin=dict(
                    l=20,
                    r=20,
                    t=65,
                    b=20
                ),
                hovermode="x unified",
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"static_info_curve_{battery}"
            )

            with st.expander(
                "View NASA Experimental Measurements",
                expanded=False
            ):

                display_df = battery_df[
                    [
                        "battery",
                        "discharge_idx",
                        "capacity_Ah",
                        "ambient_temperature_C"
                    ]
                ].rename(
                    columns={
                        "battery": "Battery Cell",
                        "discharge_idx": "Discharge Cycle",
                        "capacity_Ah": "Capacity (Ah)",
                        "ambient_temperature_C": "Temperature (°C)"
                    }
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                f"NASA dataset capacity range: "
                f"{lowest_capacity:.3f}–{highest_capacity:.3f} Ah"
            )

    st.divider()

    st.caption(
        "Data source: NASA PCoE Battery Data Set. "
        "This page displays original experimental dataset values only. "
        "Saved live measurements do not modify this information."
    )


# =========================================================
# BATTERY ANALYSIS
# =========================================================

def battery_analysis():

    st.title("Battery Analysis")

    st.caption(
        "Examine individual battery-cell performance, "
        "capacity degradation and State of Health."
    )

    df = load_battery_data()

    if df is None:
        st.error("Battery dataset could not be loaded.")
        return

    battery = st.selectbox(
        "Select Battery Cell",
        BATTERIES,
        key="analysis_battery"
    )

    battery_df = df[
        df["battery"] == battery
    ].sort_values("discharge_idx").copy()

    if battery_df.empty:
        st.warning("No records available for this battery.")
        return

    measurement = get_cell_measurement(df, battery)

    capacity = measurement["capacity"]
    cycles = measurement["cycle"]

    soh = capacity / RATED_CAPACITY * 100
    status = measurement["status"]

    st.divider()

    st.markdown(f"### {battery} · Current Battery Condition")

    st.markdown(
        get_status_html(status),
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Measured Capacity", f"{capacity:.3f} Ah")
    col2.metric("State of Health", f"{soh:.1f}%")

    col3.metric(
        "Discharge Cycle",
        cycles if cycles is not None else "N/A"
    )

    col4.metric("EOL Threshold", f"{EOL_CAPACITY:.2f} Ah")

    st.caption(f"Measurement source: {measurement['source']}")

    if status == "Critical":

        st.error(
            "CRITICAL — Battery has reached or crossed "
            "its end-of-life threshold."
        )

    elif status == "Moderate":

        st.warning(
            "MODERATE — Battery capacity has entered "
            "the warning range."
        )

    elif status == "Anomaly":

        st.warning(
            "ANOMALY — Unusual battery capacity behaviour detected."
        )

    else:

        st.success(
            "GOOD — Battery capacity is currently above "
            "the warning range."
        )

    st.divider()

    # =====================================================
    # GRAPH 1: CAPACITY DEGRADATION
    # =====================================================

    st.subheader("Capacity Degradation")

    fig = px.line(
        battery_df,
        x="discharge_idx",
        y="capacity_Ah",
        markers=True,
        labels={
            "discharge_idx": "Discharge Cycle",
            "capacity_Ah": "Capacity (Ah)"
        },
        title=f"{battery} · Discharge Capacity Trend"
    )

    fig.update_traces(
        line=dict(color="#B69AFF", width=3),
        marker=dict(size=6)
    )

    fig.add_hline(
        y=RATED_CAPACITY,
        line_dash="dash",
        line_color="#91BDF5",
        annotation_text="Rated Capacity (2.0 Ah)"
    )

    fig.add_hline(
        y=WARNING_CAPACITY,
        line_dash="dash",
        line_color="#E6A4D7",
        annotation_text="Warning (1.8 Ah)"
    )

    fig.add_hline(
        y=EOL_CAPACITY,
        line_dash="dash",
        line_color="#F080A8",
        annotation_text="End of Life (1.5 Ah)"
    )

    if cycles is not None:

        fig.add_scatter(
            x=[cycles],
            y=[capacity],
            mode="markers",
            marker=dict(
                size=14,
                color="white",
                line=dict(width=2, color="#B69AFF")
            ),
            name="Current measurement"
        )

    fig.update_layout(
        template="plotly_dark",
        height=430
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # GRAPH 2: STATE OF HEALTH TREND
    # =====================================================

    st.subheader("State of Health Trend")

    soh_df = battery_df.copy()

    soh_df["SOH (%)"] = (
        soh_df["capacity_Ah"] / RATED_CAPACITY
    ) * 100

    soh_fig = px.line(
        soh_df,
        x="discharge_idx",
        y="SOH (%)",
        markers=True,
        labels={
            "discharge_idx": "Discharge Cycle",
            "SOH (%)": "State of Health (%)"
        },
        title=f"{battery} · SOH Trend"
    )

    soh_fig.update_traces(
        line=dict(color="#E6A4D7", width=3),
        marker=dict(size=6)
    )

    soh_fig.add_hline(
        y=(EOL_CAPACITY / RATED_CAPACITY) * 100,
        line_dash="dash",
        line_color="#F080A8",
        annotation_text="End-of-Life SOH"
    )

    soh_fig.update_layout(
        template="plotly_dark",
        height=380
    )

    st.plotly_chart(soh_fig, use_container_width=True)

    # =====================================================
    # GRAPH 3: ACTUAL VS PREDICTED CAPACITY
    # =====================================================

    st.divider()

    st.subheader("Actual vs Predicted Capacity")

    st.caption(
        "Comparison of NASA-measured battery capacity "
        "against capacity estimates from a linear regression model."
    )

    x = battery_df["discharge_idx"].to_numpy(dtype=float)
    y = battery_df["capacity_Ah"].to_numpy(dtype=float)

    if len(x) < 2:

        st.warning(
            "At least two discharge-cycle measurements are "
            "required to generate capacity predictions."
        )

    else:

        # Fit linear regression using the selected cell's data.
        slope, intercept = np.polyfit(x, y, 1)

        predicted_capacity = slope * x + intercept

        prediction_df = pd.DataFrame({
            "Discharge Cycle": x,
            "Actual Capacity": y,
            "Predicted Capacity": predicted_capacity
        })

        prediction_fig = px.line(
            prediction_df,
            x="Discharge Cycle",
            y="Actual Capacity",
            markers=True,
            title=f"{battery} · Actual vs Predicted Capacity"
        )

        prediction_fig.update_traces(
            name="Actual Capacity",
            line=dict(
                color="#B69AFF",
                width=3
            ),
            marker=dict(size=6)
        )

        prediction_fig.add_scatter(
            x=x,
            y=predicted_capacity,
            mode="lines",
            name="Predicted Capacity",
            line=dict(
                color="#91BDF5",
                width=3,
                dash="dash"
            )
        )

        prediction_fig.add_hline(
            y=WARNING_CAPACITY,
            line_dash="dash",
            line_color="#E6A4D7",
            annotation_text="Warning (1.8 Ah)"
        )

        prediction_fig.add_hline(
            y=EOL_CAPACITY,
            line_dash="dash",
            line_color="#F080A8",
            annotation_text="End of Life (1.5 Ah)"
        )

        prediction_fig.update_layout(
            template="plotly_dark",
            height=430,
            xaxis_title="Discharge Cycle",
            yaxis_title="Capacity (Ah)",
            hovermode="x unified"
        )

        st.plotly_chart(
            prediction_fig,
            use_container_width=True
        )

        # Prediction model performance on the fitted dataset.
        mae = float(np.mean(np.abs(y - predicted_capacity)))

        rmse = float(
            np.sqrt(np.mean((y - predicted_capacity) ** 2))
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Prediction MAE",
            f"{mae:.4f} Ah"
        )

        c2.metric(
            "Prediction RMSE",
            f"{rmse:.4f} Ah"
        )

        c3.metric(
            "Model",
            "Linear Regression"
        )

        st.caption(
            "Note: These predictions are fitted to the same "
            "historical measurements shown in the graph. "
            "They illustrate the regression trend and are not "
            "independent test-set predictions or a validated "
            "future-capacity forecast."
        )

    # =====================================================
    # CAPACITY SUMMARY
    # =====================================================

    first_capacity = float(
        battery_df.iloc[0]["capacity_Ah"]
    )

    capacity_reduction = first_capacity - capacity

    st.subheader("Capacity Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Initial Dataset Capacity",
        f"{first_capacity:.3f} Ah"
    )

    col2.metric(
        "Capacity Reduction",
        f"{capacity_reduction:.3f} Ah"
    )

    col3.metric(
        "Recorded Dataset Measurements",
        len(battery_df)
    )

    with st.expander("View Recent Dataset Measurements"):

        st.dataframe(
            battery_df.tail(10).sort_values(
                "discharge_idx",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# LIVE MEASUREMENT
# =========================================================

def live_measurement(df):

    st.subheader("Live Measurement")

    st.caption(
        "Enter a new battery capacity measurement to evaluate "
        "its condition and estimate its remaining useful life."
    )

    st.info(
        "Capacity ranges: Good: above 1800 mAh · "
        "Moderate: 1500–1800 mAh · Critical: 1500 mAh or below."
    )

    with st.form("aurora_live_measurement_form"):

        cell = st.selectbox(
            "Select Battery Cell",
            BATTERIES,
            key="live_cell"
        )

        capacity_mAh = st.number_input(
            "Measured Capacity (mAh)",
            min_value=1.0,
            max_value=10000.0,
            value=1900.0,
            step=10.0,
            help="Enter the measured discharge capacity in mAh."
        )

        internal_resistance = st.number_input(
            "Internal Resistance (mΩ) — Optional",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=0.1,
            help="Enter the measured internal resistance, if available."
        )

        submitted = st.form_submit_button(
            "Analyse Measurement",
            use_container_width=True
        )

    if not submitted:
        return

    capacity = capacity_mAh / 1000.0

    battery_df = df[
        df["battery"] == cell
    ].sort_values("discharge_idx")

    if battery_df.empty:
        st.error("No recorded data available for this cell.")
        return

    soh = capacity / RATED_CAPACITY * 100

    x = battery_df["discharge_idx"].to_numpy(dtype=float)
    y = battery_df["capacity_Ah"].to_numpy(dtype=float)

    if len(x) >= 2:

        slope, intercept = np.polyfit(x, y, 1)

        if abs(slope) > 1e-10:

            effective_cycle = (capacity - intercept) / slope

        else:

            effective_cycle = float(x[-1])

        effective_cycle = float(
            np.clip(effective_cycle, np.min(x), np.max(x))
        )

    else:

        slope = 0
        intercept = capacity
        effective_cycle = float(x[-1])

    if slope < 0:

        eol_cycle = (EOL_THRESHOLD - intercept) / slope
        rul = max(0, eol_cycle - effective_cycle)

    else:

        rul = None

    if capacity <= EOL_THRESHOLD:
        verdict = "CRITICAL"

    elif capacity <= WARNING_THRESHOLD:
        verdict = "WARNING"

    else:
        verdict = "HEALTHY"

    flags = []

    mean_capacity = float(np.mean(y))
    std_capacity = float(np.std(y))

    if std_capacity > 1e-10:

        z_score = (capacity - mean_capacity) / std_capacity

        if abs(z_score) > 3:

            flags.append(
                f"Capacity outlier detected (z-score: {z_score:.2f})."
            )

    previous_reading = get_previous_reading(cell)

    if previous_reading is not None:

        previous_capacity = float(previous_reading["capacity"])

        if previous_capacity > 0:

            capacity_drop = (
                previous_capacity - capacity
            ) / previous_capacity

            if capacity_drop > 0.03:

                flags.append(
                    "Capacity dropped by more than 3% "
                    "compared with the previous saved reading."
                )

    if internal_resistance > IR_WARNING_THRESHOLD:

        flags.append(
            f"High internal resistance detected "
            f"({internal_resistance:.1f} mΩ)."
        )

    if flags and verdict == "HEALTHY":
        verdict = "ANOMALY"

    saved = save_reading(
        cell=cell,
        capacity=capacity,
        cycle=effective_cycle,
        verdict=verdict,
        internal_resistance=(
            internal_resistance
            if internal_resistance > 0
            else None
        ),
        mode="Live measurement",
        flags=flags
    )

    st.session_state["aurora_last_live_result"] = {
        "cell": cell,
        "capacity": capacity,
        "capacity_mAh": capacity_mAh,
        "soh": soh,
        "effective_cycle": effective_cycle,
        "rul": rul,
        "verdict": verdict,
        "flags": flags,
        "internal_resistance": internal_resistance,
        "timestamp": saved["timestamp"]
    }

    st.rerun()


# =========================================================
# DISPLAY LIVE MEASUREMENT RESULTS
# =========================================================

def show_live_result():

    result = st.session_state.get("aurora_last_live_result")

    if result is None:
        return

    st.divider()

    st.subheader(
        f"Measurement Results · {result['cell']}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Measured Capacity",
        f"{result['capacity_mAh']:.0f} mAh"
    )

    c2.metric(
        "State of Health",
        f"{result['soh']:.1f}%"
    )

    c3.metric(
        "Effective Cycle",
        f"{result['effective_cycle']:.1f}"
    )

    if result["rul"] is not None:

        c4.metric(
            "Estimated Remaining Life",
            f"{result['rul']:.0f} cycles"
        )

    else:

        c4.metric(
            "Estimated Remaining Life",
            "Not estimable"
        )

    verdict = result["verdict"]

    if verdict == "CRITICAL":

        st.error(
            f"CRITICAL CONDITION — {result['cell']} "
            "has reached or crossed the end-of-life threshold."
        )

    elif verdict == "WARNING":

        st.warning(
            f"WARNING — {result['cell']} "
            "has entered the capacity warning range."
        )

    elif verdict == "ANOMALY":

        st.warning(
            f"ANOMALY DETECTED — {result['cell']} "
            "has unusual capacity behaviour."
        )

    else:

        st.success(
            f"GOOD CONDITION — {result['cell']} "
            "is within the normal capacity range."
        )

    if result["internal_resistance"] > 0:

        st.metric(
            "Internal Resistance",
            f"{result['internal_resistance']:.1f} mΩ"
        )

    if result["flags"]:

        st.markdown("**Diagnostic Flags**")

        for flag in result["flags"]:
            st.warning(flag)

    st.success(
        "Measurement saved successfully. "
        "Battery Analysis and Alerts now use this measurement."
    )

    st.caption(f"Saved at: {result['timestamp']}")


# =========================================================
# FAULT DETECTION PAGE
# =========================================================

def fault_detection():

    st.title("Fault Detection")

    st.caption(
        "Analyse a new battery measurement and evaluate "
        "its current condition."
    )

    df = load_battery_data()

    if df is None:
        st.error("Battery dataset could not be loaded.")
        return

    live_measurement(df)
    show_live_result()


# =========================================================
# ALERTS
# =========================================================

def alerts():

    st.title("Battery Alerts")

    df = load_battery_data()

    if df is None:
        st.error("Battery dataset could not be loaded.")
        return

    summary = get_individual_cycle_summary(df)

    if summary.empty:
        st.warning("No battery measurements are available.")
        return

    st.write(
        "Alerts use the latest saved live measurement for each cell. "
        "Cells without a saved live measurement use their latest "
        "NASA experimental dataset value."
    )

    critical = summary[summary["Status"] == "Critical"]
    moderate = summary[summary["Status"] == "Moderate"]
    anomaly = summary[summary["Status"] == "Anomaly"]
    good = summary[summary["Status"] == "Good"]

    st.divider()

    st.markdown("### System Alert Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Critical Cells", len(critical))
    c2.metric("Warning Cells", len(moderate))
    c3.metric("Anomalies", len(anomaly))
    c4.metric("Good Cells", len(good))

    if not critical.empty:

        st.error("CRITICAL BATTERY ALERTS")

        for _, r in critical.iterrows():

            st.markdown(
                f"**{r['Battery']} — Cycle {r['Selected Cycle']}**  \n"
                f"Capacity: **{r['Capacity (Ah)']:.3f} Ah** · "
                f"SOH: **{r['SOH (%)']:.1f}%** · "
                "End-of-life threshold reached or crossed."
            )

    if not moderate.empty:

        st.warning("BATTERY WARNINGS")

        for _, r in moderate.iterrows():

            st.markdown(
                f"**{r['Battery']} — Cycle {r['Selected Cycle']}**  \n"
                f"Capacity: **{r['Capacity (Ah)']:.3f} Ah** · "
                f"SOH: **{r['SOH (%)']:.1f}%** · "
                "Capacity has entered the warning range."
            )

    if not anomaly.empty:

        st.warning("BATTERY ANOMALIES")

        for _, r in anomaly.iterrows():

            st.markdown(
                f"**{r['Battery']} — Cycle {r['Selected Cycle']}**  \n"
                f"Capacity: **{r['Capacity (Ah)']:.3f} Ah** · "
                f"SOH: **{r['SOH (%)']:.1f}%** · "
                "Unusual capacity behaviour detected."
            )

    if not good.empty:

        st.success("GOOD CONDITION")

        st.write(
            ", ".join(
                f"{r['Battery']} (cycle {r['Selected Cycle']})"
                for _, r in good.iterrows()
            )
        )

    st.divider()

    st.markdown("### Battery-wise Alert Status")

    display = summary.copy()

    display["Capacity (Ah)"] = display["Capacity (Ah)"].round(3)
    display["SOH (%)"] = display["SOH (%)"].round(2)

    display = display.sort_values("Battery")

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# NAVIGATION PAGES
# =========================================================

dashboard_page = st.Page(
    dashboard,
    title="Dashboard",
    icon=":material/dashboard:",
    default=True
)

information_page = st.Page(
    battery_information,
    title="Battery Information",
    icon=":material/info:"
)

fault_page = st.Page(
    fault_detection,
    title="Fault Detection",
    icon=":material/troubleshoot:"
)

alerts_page = st.Page(
    alerts,
    title="Alerts",
    icon=":material/notifications_active:"
)

analysis_page = st.Page(
    battery_analysis,
    title="Battery Analysis",
    icon=":material/battery_charging_full:"
)


# =========================================================
# PROJECT LAB PAGES
# =========================================================

eda_page = st.Page(
    "pages/02_Exploratory_Data_Analysis.py",
    title="Exploratory Data Analysis",
    icon=":material/analytics:"
)

cleaning_page = st.Page(
    "pages/03_Data_Cleaning_Preprocessing.py",
    title="Data Cleaning",
    icon=":material/cleaning_services:"
)

preprocessing_demo_page = st.Page(
    "pages/99_Preprocessing_Demonstration.py",
    title="Preprocessing Demonstration",
    icon=":material/science:"
)

features_page = st.Page(
    "pages/04_Feature_Engineering.py",
    title="Feature Engineering",
    icon=":material/auto_awesome:"
)

prediction_page = st.Page(
    "pages/05_Predictive_Modelling.py",
    title="Predictive Modelling",
    icon=":material/psychology:"
)


# =========================================================
# NAVIGATION SETUP
# =========================================================

all_pages = [
    dashboard_page,
    information_page,
    fault_page,
    alerts_page,
    analysis_page,
    eda_page,
    cleaning_page,
    preprocessing_demo_page,
    features_page,
    prediction_page
]

navigation = st.navigation(
    all_pages,
    position="hidden"
)


# =========================================================
# CUSTOM SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("---")

    st.markdown(
        '<div class="nav-section-label">DASHBOARD</div>',
        unsafe_allow_html=True
    )

    st.page_link(
        dashboard_page,
        label="Home",
        icon=":material/home:"
    )

    st.page_link(
        information_page,
        label="Battery Information",
        icon=":material/info:"
    )

    st.page_link(
        fault_page,
        label="Fault Detection",
        icon=":material/troubleshoot:"
    )

    sidebar_df = load_battery_data()

    alert_label = "Alerts"

    if sidebar_df is not None:

        sidebar_summary = get_individual_cycle_summary(
            sidebar_df
        )

        alert_level = get_alert_level(sidebar_summary)

        if alert_level == "critical":
            alert_label = "Alerts 🔴"

        elif alert_level in ["moderate", "anomaly"]:
            alert_label = "Alerts ⚠️"

    st.page_link(
        alerts_page,
        label=alert_label,
        icon=":material/notifications_active:"
    )

    st.page_link(
        analysis_page,
        label="Battery Analysis",
        icon=":material/battery_charging_full:"
    )

    st.markdown("---")

    with st.expander("PROJECT LAB", expanded=False):

        st.page_link(
            eda_page,
            label="Exploratory Data Analysis"
        )

        st.page_link(
            cleaning_page,
            label="Data Cleaning"
        )

        st.page_link(
            preprocessing_demo_page,
            label="Preprocessing Demonstration"
        )

        st.page_link(
            features_page,
            label="Feature Engineering"
        )

        st.page_link(
            prediction_page,
            label="Predictive Modelling"
        )


# =========================================================
# RUN APPLICATION
# =========================================================

navigation.run()
