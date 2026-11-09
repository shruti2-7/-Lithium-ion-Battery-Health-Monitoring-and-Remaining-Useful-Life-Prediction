
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# AURORA — FEATURE ENGINEERING
# =========================================================

st.set_page_config(
    page_title="Feature Engineering | AURORA",
    page_icon="",
    layout="wide"
)

DATA_PATH = Path("data/processed/discharge_capacity.csv")

RATED_CAPACITY = 2.0
EOL_CAPACITY = 1.5

TRAIN_BATTERIES = ["B0005", "B0006", "B0007"]
TEST_BATTERY = "B0018"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    required = [
        "battery",
        "record_idx",
        "discharge_idx",
        "capacity_Ah",
        "ambient_temperature_C"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()

    df["battery"] = df["battery"].astype(str)

    df = df.sort_values(
        ["battery", "discharge_idx"]
    ).reset_index(drop=True)

    return df


try:
    original_df = load_data()
except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()


# =========================================================
# FEATURE CREATION
# =========================================================

def create_features(data):

    df = data.copy()

    grouped = df.groupby("battery", sort=False)

    # State of Health
    df["SOH_percent"] = (
        df["capacity_Ah"] / RATED_CAPACITY
    ) * 100

    # Previous cycle capacity
    df["previous_capacity_Ah"] = (
        grouped["capacity_Ah"].shift(1)
    )

    # Capacity change
    df["capacity_change_Ah"] = (
        df["capacity_Ah"] -
        df["previous_capacity_Ah"]
    )

    # Difference between discharge cycle indices
    df["cycle_interval"] = (
        grouped["discharge_idx"].diff()
    )

    # Capacity fade rate
    df["capacity_fade_rate_Ah_per_cycle"] = (
        -df["capacity_change_Ah"] /
        df["cycle_interval"].replace(0, np.nan)
    )

    # Previous five-cycle rolling mean
    df["rolling_mean_previous_5_Ah"] = (
        df.groupby("battery")["capacity_Ah"]
        .transform(
            lambda s: s.shift(1)
            .rolling(5, min_periods=5)
            .mean()
        )
    )

    # Previous five-cycle rolling standard deviation
    df["rolling_std_previous_5_Ah"] = (
        df.groupby("battery")["capacity_Ah"]
        .transform(
            lambda s: s.shift(1)
            .rolling(5, min_periods=5)
            .std(ddof=1)
        )
    )

    # Distance from EOL threshold
    df["capacity_above_EOL_Ah"] = (
        df["capacity_Ah"] - EOL_CAPACITY
    )

    # Normalized cycle progression
    df["cycle_progress_percent"] = (
        df.groupby("battery")["discharge_idx"]
        .transform(
            lambda s: (
                (s - s.min()) / (s.max() - s.min()) * 100
                if s.max() != s.min()
                else 0
            )
        )
    )

    # Historical-only features for future prediction
    df["previous_capacity_change_Ah"] = (
        df.groupby("battery")["capacity_change_Ah"]
        .shift(1)
    )

    df["previous_fade_rate_Ah_per_cycle"] = (
        df.groupby("battery")[
            "capacity_fade_rate_Ah_per_cycle"
        ].shift(1)
    )

    return df


features_df = create_features(original_df)


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Feature Engineering")
st.caption("AURORA  ·  BATTERY ANALYTICS")

st.markdown(
    "Transforming raw battery measurements into meaningful "
    "indicators for degradation analysis and future-cycle prediction."
)

st.divider()


# =========================================================
# 01 — FEATURE CREATION
# =========================================================

st.header("01 — Feature Creation")

st.write(
    "The NASA dataset provides five original input columns. "
    "Additional battery-health indicators are derived from these measurements."
)

# Original inputs
with st.expander("View original input features", expanded=False):

    st.dataframe(
        original_df.head(5),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"{len(original_df):,} records · "
        f"{len(original_df.columns)} original columns"
    )


# Compact feature cards
st.subheader("Newly derived battery-health features")

feature_cards = [
    {
        "title": "Health Indicators",
        "features": [
            ("SOH (%)", "Percentage of rated battery capacity"),
            ("Capacity above EOL", "Distance from the 1.5 Ah threshold")
        ]
    },
    {
        "title": "Capacity Degradation",
        "features": [
            ("Capacity change", "Difference from the previous cycle"),
            ("Fade rate", "Capacity loss per discharge cycle")
        ]
    },
    {
        "title": "Historical Indicators",
        "features": [
            ("Previous capacity", "Capacity measured in the previous cycle"),
            ("Rolling mean", "Average of the previous five cycles"),
            ("Rolling deviation", "Variation across the previous five cycles")
        ]
    },
    {
        "title": "Cycle Information",
        "features": [
            ("Cycle interval", "Difference between discharge indices"),
            ("Cycle progression", "Normalized position in cycle history")
        ]
    }
]

for start in range(0, len(feature_cards), 2):

    col1, col2 = st.columns(2)

    for col, card in zip(
        [col1, col2],
        feature_cards[start:start + 2]
    ):

        with col:
            with st.container(border=True):

                st.markdown(f"**{card['title']}**")

                for feature_name, description in card["features"]:

                    st.markdown(
                        f"**{feature_name}**"
                    )

                    st.caption(description)


# Small preview
st.markdown("**Generated feature preview**")

preview_columns = [
    "battery",
    "discharge_idx",
    "SOH_percent",
    "capacity_change_Ah",
    "capacity_fade_rate_Ah_per_cycle",
    "rolling_mean_previous_5_Ah"
]

st.dataframe(
    features_df[preview_columns].head(5).round(4),
    use_container_width=True,
    hide_index=True,
    height=210
)

st.caption(
    "Current-cycle capacity-derived features are descriptive. "
    "They must not be used to predict that same cycle's capacity."
)


# =========================================================
# 02 — MATHEMATICAL FORMULATION
# =========================================================

st.divider()
st.header("02 — Mathematical Formulation")

st.write(
    "The following equations describe how each battery-health "
    "indicator is calculated from the original measurements."
)

formula_rows = [
    (
        "State of Health (SOH)",
        r"SOH_i=\frac{Q_i}{Q_{\mathrm{rated}}}\times100",
        [
            r"$Q_i$ = Capacity at the current discharge cycle.",
            r"$Q_{\mathrm{rated}}=2.0\,Ah$ = Rated battery capacity."
        ]
    ),
    (
        "Capacity Change",
        r"\Delta Q_i=Q_i-Q_{i-1}",
        [
            r"$Q_i$ = Capacity at the current discharge cycle.",
            r"$Q_{i-1}$ = Capacity at the previous discharge cycle."
        ]
    ),
    (
        "Capacity Fade Rate",
        r"F_i=-\frac{\Delta Q_i}{\Delta N_i}",
        [
            r"$\Delta Q_i$ = Change in capacity.",
            r"$\Delta N_i$ = Difference in discharge-cycle indices."
        ]
    ),
    (
        "Rolling Mean",
        r"\mu_i=\frac{1}{5}\sum_{j=1}^{5}Q_{i-j}",
        [
            r"$Q_{i-j}$ = Capacity of each of the previous five cycles.",
            r"$\mu_i$ = Mean capacity of those five cycles."
        ]
    ),
    (
        "Rolling Standard Deviation",
        r"\sigma_i=\sqrt{\frac{1}{4}\sum_{j=1}^{5}(Q_{i-j}-\mu_i)^2}",
        [
            r"$Q_{i-j}$ = Previous five cycle capacities.",
            r"$\mu_i$ = Mean of the previous five capacities.",
            r"$4$ = Degrees of freedom for sample standard deviation."
        ]
    ),
    (
        "Distance from EOL",
        r"D_i=Q_i-Q_{\mathrm{EOL}}",
        [
            r"$Q_i$ = Current cycle capacity.",
            r"$Q_{\mathrm{EOL}}=1.5\,Ah$ = End-of-life threshold."
        ]
    ),
    (
        "Cycle Progression",
        r"P_i=\frac{N_i-N_{\min}}{N_{\max}-N_{\min}}\times100",
        [
            r"$N_i$ = Current discharge-cycle index.",
            r"$N_{\min}$ = First discharge-cycle index.",
            r"$N_{\max}$ = Last discharge-cycle index."
        ]
    )
]

for name, formula, meanings in formula_rows:

    with st.container(border=True):

        st.markdown(f"**{name}**")

        formula_col, meaning_col = st.columns([1, 1.35])

        with formula_col:
            st.latex(formula)

        with meaning_col:
            for meaning in meanings:
                st.markdown(meaning)


# Calculation demonstration
st.subheader("Actual calculation from NASA data")

selected_battery = st.selectbox(
    "Choose a battery",
    sorted(features_df["battery"].unique()),
    key="calculation_battery"
)

battery_data = features_df[
    features_df["battery"] == selected_battery
].dropna(
    subset=["previous_capacity_Ah"]
).reset_index(drop=True)

if not battery_data.empty:

    row = battery_data.iloc[min(5, len(battery_data) - 1)]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current capacity",
        f"{row['capacity_Ah']:.4f} Ah"
    )

    col2.metric(
        "Previous capacity",
        f"{row['previous_capacity_Ah']:.4f} Ah"
    )

    col3.metric(
        "SOH",
        f"{row['SOH_percent']:.2f}%"
    )

    with st.expander("See calculation steps"):

        st.markdown("**1. State of Health**")

        st.latex(
            rf"SOH=\frac{{{row['capacity_Ah']:.4f}}}"
            rf"{{2.0}}\times100"
        )

        st.latex(
            rf"SOH={row['SOH_percent']:.2f}\%"
        )

        st.markdown("**2. Capacity Change**")

        st.latex(
            rf"\Delta Q={row['capacity_Ah']:.4f}"
            rf"-{row['previous_capacity_Ah']:.4f}"
            rf"={row['capacity_change_Ah']:.4f}\ Ah"
        )

        st.markdown("**3. Capacity Fade Rate**")

        st.latex(
            rf"F=-\frac{{{row['capacity_change_Ah']:.4f}}}"
            rf"{{{row['cycle_interval']:.0f}}}"
            rf"={row['capacity_fade_rate_Ah_per_cycle']:.5f}"
            rf"\ Ah/cycle"
        )


# =========================================================
# 03 — FEATURE SELECTION
# =========================================================

st.divider()
st.header("03 — Feature Selection")

st.write(
    "Feature selection identifies useful historical measurements "
    "for predicting capacity in a future discharge cycle."
)

st.caption(
    "Training batteries: B0005, B0006 and B0007 · "
    "Held-out battery: B0018"
)

candidate_features = [
    "previous_capacity_Ah",
    "previous_capacity_change_Ah",
    "previous_fade_rate_Ah_per_cycle",
    "rolling_mean_previous_5_Ah",
    "rolling_std_previous_5_Ah",
    "ambient_temperature_C",
    "cycle_progress_percent"
]

training_data = features_df[
    features_df["battery"].isin(TRAIN_BATTERIES)
].copy()

correlation_data = training_data[
    candidate_features + ["capacity_Ah"]
].replace([np.inf, -np.inf], np.nan)

correlations = (
    correlation_data.corr(numeric_only=True)["capacity_Ah"]
    .drop("capacity_Ah")
    .dropna()
    .sort_values(key=abs, ascending=False)
)

corr_df = correlations.reset_index()
corr_df.columns = ["Feature", "Correlation"]

st.markdown("**Correlation with current-cycle capacity**")

fig_corr = px.bar(
    corr_df,
    x="Correlation",
    y="Feature",
    orientation="h",
    color="Correlation",
    color_continuous_scale="Teal"
)

fig_corr.update_layout(
    height=340,
    showlegend=False,
    margin=dict(l=10, r=10, t=20, b=20)
)

st.plotly_chart(fig_corr, use_container_width=True)

st.caption(
    "Correlation is calculated on training batteries only. "
    "It measures association, not causation."
)

with st.expander("Why are these features selected?"):

    reasons = [
        ("Previous capacity", "Represents recent battery health."),
        ("Previous capacity change", "Shows the recent capacity trend."),
        ("Previous fade rate", "Describes historical capacity loss per cycle."),
        ("Rolling mean", "Smooths short-term capacity fluctuations."),
        ("Rolling standard deviation", "Measures variation in recent capacity."),
        ("Ambient temperature", "Provides operating-condition information."),
        ("Cycle progression", "Represents the battery's position in its cycle history.")
    ]

    for name, reason in reasons:
        st.markdown(f"**{name}:** {reason}")

st.warning(
    "Current-cycle capacity, SOH, current capacity change, current fade rate "
    "and distance from EOL are excluded from future-capacity prediction "
    "because they depend on the target measurement."
)


# =========================================================
# 04 — DIMENSIONALITY REDUCTION USING PCA
# =========================================================

st.divider()
st.header("04 — Dimensionality Reduction (PCA)")

st.write(
    "Principal Component Analysis (PCA) transforms correlated features "
    "into fewer uncorrelated components while retaining as much "
    "information (variance) as possible."
)

with st.expander("How does PCA work?"):

    st.latex(r"Z=XW")

    st.markdown("""
    - \(X\): Standardized feature matrix.
    - \(W\): Principal component loading matrix.
    - \(Z\): Transformed dataset with fewer dimensions.
    """)

st.markdown(
    f"**Original prediction features:** {len(candidate_features)}"
)

# Fit imputer, scaler and PCA on training batteries only
pca_train = training_data[candidate_features].replace(
    [np.inf, -np.inf],
    np.nan
)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

train_imputed = imputer.fit_transform(pca_train)
train_scaled = scaler.fit_transform(train_imputed)

pca = PCA(
    n_components=0.95,
    svd_solver="full"
)

train_pca = pca.fit_transform(train_scaled)

component_count = pca.n_components_

# Transform held-out battery using fitted training objects
test_data = features_df[
    features_df["battery"] == TEST_BATTERY
].copy()

test_input = test_data[candidate_features].replace(
    [np.inf, -np.inf],
    np.nan
)

test_imputed = imputer.transform(test_input)
test_scaled = scaler.transform(test_imputed)
test_pca = pca.transform(test_scaled)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Before PCA",
        f"{len(candidate_features)} features"
    )

with col2:
    st.metric(
        "After PCA",
        f"{component_count} components"
    )

st.caption(
    "PCA automatically retains enough components to explain at least 95% "
    "of the variance in the training data."
)

variance_df = pd.DataFrame({
    "Component": [
        f"PC{i + 1}" for i in range(component_count)
    ],
    "Individual variance (%)": (
        pca.explained_variance_ratio_ * 100
    ),
    "Cumulative variance (%)": (
        np.cumsum(pca.explained_variance_ratio_) * 100
    )
})

st.markdown("**Explained variance**")

fig_pca = go.Figure()

fig_pca.add_trace(
    go.Bar(
        x=variance_df["Component"],
        y=variance_df["Individual variance (%)"],
        name="Individual variance"
    )
)

fig_pca.add_trace(
    go.Scatter(
        x=variance_df["Component"],
        y=variance_df["Cumulative variance (%)"],
        name="Cumulative variance",
        mode="lines+markers",
        yaxis="y2"
    )
)

fig_pca.update_layout(
    height=350,
    yaxis_title="Individual variance (%)",
    yaxis2=dict(
        title="Cumulative variance (%)",
        overlaying="y",
        side="right",
        range=[0, 105]
    ),
    margin=dict(l=10, r=10, t=30, b=20)
)

st.plotly_chart(fig_pca, use_container_width=True)

with st.expander("View PCA variance values"):
    st.dataframe(
        variance_df.round(3),
        use_container_width=True,
        hide_index=True
    )

st.success(
    f"Dimensionality reduced from {len(candidate_features)} "
    f"to {component_count} components, retaining "
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}% "
    f"of training-data variance."
)

st.caption(
    "Imputation, scaling and PCA were fitted using B0005, B0006 and B0007. "
    "B0018 was transformed without being used to fit these steps."
)


# =========================================================
# 05 — FINAL FEATURE DATASET
# =========================================================

st.divider()
st.header("05 — Final Feature Dataset")

st.write(
    "The final engineered dataset combines the original measurements "
    "with the newly derived battery-health indicators."
)

final_columns = [
    "battery",
    "discharge_idx",
    "capacity_Ah",
    "ambient_temperature_C",
    "SOH_percent",
    "previous_capacity_Ah",
    "capacity_change_Ah",
    "capacity_fade_rate_Ah_per_cycle",
    "rolling_mean_previous_5_Ah",
    "rolling_std_previous_5_Ah",
    "capacity_above_EOL_Ah",
    "cycle_progress_percent"
]

final_df = features_df[final_columns].copy()

col1, col2, col3 = st.columns(3)

col1.metric("Original columns", len(original_df.columns))
col2.metric("Final columns", len(final_df.columns))
col3.metric("Total records", f"{len(final_df):,}")

st.markdown("**Final dataset preview**")

st.dataframe(
    final_df.head(8).round(4),
    use_container_width=True,
    hide_index=True,
    height=280
)

st.caption(
    "Initial rows contain unavailable historical values because "
    "previous-cycle calculations require earlier observations."
)

st.download_button(
    "Download engineered feature dataset",
    data=final_df.to_csv(index=False).encode("utf-8"),
    file_name="aurora_engineered_features.csv",
    mime="text/csv"
)

# PCA transformed training data
pca_columns = [
    f"PC{i + 1}" for i in range(component_count)
]

pca_output = pd.DataFrame(
    train_pca,
    columns=pca_columns
)

pca_output.insert(
    0,
    "battery",
    training_data["battery"].values
)

pca_output.insert(
    1,
    "discharge_idx",
    training_data["discharge_idx"].values
)

st.download_button(
    "Download PCA training dataset",
    data=pca_output.to_csv(index=False).encode("utf-8"),
    file_name="aurora_pca_training_features.csv",
    mime="text/csv"
)


# =========================================================
# SUMMARY
# =========================================================

st.divider()

st.subheader("Feature Engineering Summary")

st.markdown("""
- Derived battery-health and degradation indicators.
- Applied mathematical calculations to actual NASA measurements.
- Examined historical feature relationships with capacity.
- Demonstrated dimensionality reduction using PCA.
- Prepared downloadable engineered and PCA-transformed datasets.
""")

st.info(
    "PCA is included as a separate analytical demonstration. "
    "Its components will only be used in the final prediction model "
    "if model evaluation shows that PCA is beneficial."
)
