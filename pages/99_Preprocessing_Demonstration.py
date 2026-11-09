from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AURORA | Preprocessing Demonstration",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .section-label {
        color: #36D6C5;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .main-heading {
        font-size: 2.3rem;
        font-weight: 750;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
    }

    .sub-heading {
        color: #9CAFC2;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .info-panel {
        background: rgba(30, 48, 68, 0.55);
        border: 1px solid rgba(100, 140, 170, 0.22);
        border-radius: 12px;
        padding: 17px;
        margin-bottom: 12px;
    }

    .metric-panel {
        background: rgba(24, 42, 61, 0.65);
        border: 1px solid rgba(100, 140, 170, 0.25);
        border-radius: 12px;
        padding: 15px;
    }

    .metric-label {
        color: #9CAFC2;
        font-size: 0.82rem;
    }

    .metric-value {
        color: #F1F5F9;
        font-size: 1.65rem;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: rgba(24, 42, 61, 0.55);
        border: 1px solid rgba(100, 140, 170, 0.22);
        padding: 13px;
        border-radius: 10px;
    }

    h2, h3 {
        color: #E8F0F8;
    }

    .small-note {
        color: #9CAFC2;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown('<div class="section-label">AURORA / PROJECT LAB</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="main-heading">Data Preprocessing Demonstration</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-heading">'
    'Identify data-quality issues, apply cleaning techniques, '
    'and compare outlier detection methods.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

DATA_PATH = Path("data/processed/discharge_capacity.csv")

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Expected file: "
        "data/processed/discharge_capacity.csv"
    )
    st.stop()

original_df = pd.read_csv(DATA_PATH)

# Find the capacity column
capacity_candidates = [
    "capacity_Ah",
    "capacity",
    "Capacity",
    "discharge_capacity",
    "Capacity_Ah"
]

CAPACITY_COL = next(
    (col for col in capacity_candidates if col in original_df.columns),
    None
)

if CAPACITY_COL is None:
    st.error(
        "Could not identify the capacity column. "
        "Please check the column names in your CSV."
    )
    st.write("Available columns:", list(original_df.columns))
    st.stop()

original_df[CAPACITY_COL] = pd.to_numeric(
    original_df[CAPACITY_COL],
    errors="coerce"
)

original_df = original_df.reset_index(drop=True)

valid_capacity = original_df[CAPACITY_COL].dropna()

if len(valid_capacity) < 10:
    st.error("Not enough valid capacity measurements for this demonstration.")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_iqr_limits(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper, q1, q3, iqr


def calculate_zscore_limits(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    mean = values.mean()
    std = values.std(ddof=0)

    if std == 0:
        return mean, mean, mean, std

    lower = mean - 3 * std
    upper = mean + 3 * std

    return lower, upper, mean, std


def get_outlier_mask(series, method):
    values = pd.to_numeric(series, errors="coerce")

    if method == "IQR":
        lower, upper, _, _, _ = calculate_iqr_limits(values)
        return (values < lower) | (values > upper)

    lower, upper, _, _ = calculate_zscore_limits(values)
    return (values < lower) | (values > upper)


def make_boxplot(df, column, title, highlight_mask=None):
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=valid,
            name="Capacity",
            boxpoints=False,
            marker=dict(color="#35C9C0"),
            line=dict(color="#35C9C0"),
            fillcolor="rgba(53, 201, 192, 0.20)"
        )
    )

    if highlight_mask is not None:
        highlighted = values[highlight_mask.fillna(False)]

        if len(highlighted) > 0:
            fig.add_trace(
                go.Scatter(
                    x=["Capacity"] * len(highlighted),
                    y=highlighted,
                    mode="markers",
                    name="Detected outliers",
                    marker=dict(
                        color="#FF6B75",
                        size=10,
                        symbol="circle"
                    )
                )
            )

    fig.update_layout(
        title=title,
        height=390,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14, 27, 42, 0.5)",
        showlegend=True,
        yaxis_title="Capacity (Ah)",
        xaxis_title="",
        margin=dict(l=15, r=15, t=55, b=20)
    )

    return fig


def metric_scores(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


# ============================================================
# CREATE A SEPARATE IMPERFECT DATASET
# Original NASA data is never modified.
# ============================================================

@st.cache_data
def create_imperfect_dataset(df, capacity_col):
    dirty = df.copy().reset_index(drop=True)

    # Insert three missing values
    missing_indices = [5, 20, 40]

    for idx in missing_indices:
        if idx < len(dirty):
            dirty.loc[idx, capacity_col] = np.nan

    # Insert two duplicate rows
    duplicate_indices = [10, 25]
    duplicate_rows = dirty.iloc[duplicate_indices].copy()

    dirty = pd.concat(
        [dirty, duplicate_rows],
        ignore_index=True
    )

    # Insert two artificial extreme capacity values
    outlier_indices = [50, 100]

    if len(dirty) > max(outlier_indices):
        dirty.loc[outlier_indices[0], capacity_col] = 3.5
        dirty.loc[outlier_indices[1], capacity_col] = 0.2

    return dirty, missing_indices, duplicate_indices, outlier_indices


dirty_df, missing_indices, duplicate_indices, artificial_outlier_indices = (
    create_imperfect_dataset(original_df, CAPACITY_COL)
)


# ============================================================
# OVERVIEW
# ============================================================

st.markdown("## Dataset overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Original records", f"{len(original_df):,}")

with col2:
    st.metric("Demonstration records", f"{len(dirty_df):,}")

with col3:
    st.metric(
        "Missing values inserted",
        int(dirty_df[CAPACITY_COL].isna().sum())
    )

with col4:
    st.metric(
        "Duplicate rows inserted",
        int(dirty_df.duplicated().sum())
    )

st.caption(
    "The demonstration dataset contains simulated data-quality problems. "
    "The original NASA dataset is preserved."
)

st.divider()


# ============================================================
# SECTION 1 — MISSING VALUE HANDLING
# ============================================================

st.markdown('<div class="section-label">01 / DATA QUALITY</div>',
            unsafe_allow_html=True)

st.markdown("## Missing value handling")

st.write(
    "Missing values occur when a measurement is unavailable or was not "
    "recorded. Here, three missing capacity measurements are inserted "
    "into a separate copy of the dataset."
)

missing_count = dirty_df[CAPACITY_COL].isna().sum()

m1, m2, m3 = st.columns(3)

m1.metric("Missing values detected", int(missing_count))
m2.metric("Valid capacity values", int(dirty_df[CAPACITY_COL].notna().sum()))
m3.metric("Missing percentage", f"{missing_count / len(dirty_df) * 100:.2f}%")

st.markdown("### Before treatment")

missing_preview = dirty_df.loc[
    dirty_df[CAPACITY_COL].isna()
].copy()

st.dataframe(
    missing_preview,
    use_container_width=True,
    hide_index=True
)

st.markdown("### Choose a missing-value treatment")

missing_method = st.selectbox(
    "Missing value treatment",
    [
        "Median imputation",
        "Mean imputation",
        "Drop missing rows"
    ],
    key="missing_method"
)

clean_missing_df = dirty_df.copy()

if missing_method == "Median imputation":
    replacement_value = clean_missing_df[CAPACITY_COL].median()

    clean_missing_df[CAPACITY_COL] = (
        clean_missing_df[CAPACITY_COL].fillna(replacement_value)
    )

    st.info(
        f"Replacement value: **{replacement_value:.4f} Ah** "
        "(median of the available capacity measurements)."
    )

elif missing_method == "Mean imputation":
    replacement_value = clean_missing_df[CAPACITY_COL].mean()

    clean_missing_df[CAPACITY_COL] = (
        clean_missing_df[CAPACITY_COL].fillna(replacement_value)
    )

    st.info(
        f"Replacement value: **{replacement_value:.4f} Ah** "
        "(mean of the available capacity measurements)."
    )

else:
    clean_missing_df = clean_missing_df.dropna(
        subset=[CAPACITY_COL]
    ).copy()

    st.info("Rows with missing capacity measurements are removed.")

st.markdown("### Replacement details")

if missing_method != "Drop missing rows":
    replacement_details = pd.DataFrame({
        "Record index": missing_indices,
        "Original capacity": ["Missing (NaN)"] * len(missing_indices),
        "Replacement capacity (Ah)": [
            round(replacement_value, 4)
        ] * len(missing_indices)
    })

    st.dataframe(
        replacement_details,
        use_container_width=True,
        hide_index=True
    )

st.markdown("### After treatment")

remaining_missing = int(clean_missing_df[CAPACITY_COL].isna().sum())

r1, r2, r3 = st.columns(3)

r1.metric("Missing before", int(missing_count))
r2.metric("Missing after", remaining_missing)
r3.metric(
    "Rows after treatment",
    len(clean_missing_df)
)

if remaining_missing == 0:
    st.success("Verification successful: no missing capacity values remain.")
else:
    st.warning(f"{remaining_missing} missing values still remain.")

st.divider()


# ============================================================
# SECTION 2 — DUPLICATE RECORD REMOVAL
# ============================================================

st.markdown('<div class="section-label">02 / DATA QUALITY</div>',
            unsafe_allow_html=True)

st.markdown("## Duplicate record removal")

st.write(
    "Duplicate records are repeated rows that can distort statistical "
    "analysis and model training. The demonstration copy contains two "
    "inserted duplicate records."
)

duplicate_mask = dirty_df.duplicated(keep="first")
duplicate_count = int(duplicate_mask.sum())

d1, d2, d3 = st.columns(3)

d1.metric("Records before removal", len(dirty_df))
d2.metric("Duplicate records detected", duplicate_count)
d3.metric("Expected records after removal", len(dirty_df) - duplicate_count)

st.markdown("### Detected duplicate records")

if duplicate_count > 0:
    duplicate_preview = dirty_df.loc[duplicate_mask].copy()

    st.dataframe(
        duplicate_preview,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No duplicate records found.")

st.markdown("### Remove duplicates")

remove_duplicates = st.checkbox(
    "Apply duplicate removal using drop_duplicates()",
    value=True
)

if remove_duplicates:
    duplicate_clean_df = clean_missing_df.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    st.success(
        f"{duplicate_count} duplicate records removed. "
        f"{len(duplicate_clean_df)} records remain."
    )
else:
    duplicate_clean_df = clean_missing_df.copy()

    st.warning("Duplicate removal is disabled.")

d4, d5, d6 = st.columns(3)

d4.metric("Records before", len(clean_missing_df))
d5.metric("Records after", len(duplicate_clean_df))
d6.metric(
    "Duplicates remaining",
    int(duplicate_clean_df.duplicated().sum())
)

st.divider()


# ============================================================
# SECTION 3 — IQR VS Z-SCORE BENCHMARK
# ============================================================

st.markdown('<div class="section-label">03 / OUTLIER ANALYSIS</div>',
            unsafe_allow_html=True)

st.markdown("## IQR vs Z-score comparison")

st.write(
    "An outlier is an observation that differs substantially from the "
    "general pattern of the data. IQR uses quartiles, while Z-score uses "
    "the mean and standard deviation."
)

st.markdown("""
<div class="info-panel">
<b>Experiment design</b><br>
A controlled right-skewed benchmark is generated using the NASA capacity
measurements as its source. Artificial anomalies are added with known
labels. Both methods are evaluated on exactly the same observations.
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Generate a right-skewed NASA-derived reference distribution
# ------------------------------------------------------------

@st.cache_data
def create_skewed_benchmark(nasa_values):
    rng = np.random.default_rng(42)

    nasa_values = np.asarray(nasa_values, dtype=float)
    nasa_values = nasa_values[np.isfinite(nasa_values)]

    median_capacity = float(np.median(nasa_values))

    # Use resampled NASA values with a positive lognormal multiplier.
    # Increase skew if needed to create a meaningful IQR/Z-score interval.
    selected_sigma = 0.25

    for sigma in np.arange(0.20, 0.81, 0.05):
        sampled = rng.choice(
            nasa_values,
            size=500,
            replace=True
        )

        reference = sampled * rng.lognormal(
            mean=0,
            sigma=sigma,
            size=500
        )

        q1 = np.quantile(reference, 0.25)
        q3 = np.quantile(reference, 0.75)
        iqr = q3 - q1

        iqr_upper = q3 + 1.5 * iqr

        mean = np.mean(reference)
        std = np.std(reference)

        z_upper = mean + 3 * std

        if z_upper > iqr_upper + 0.03:
            selected_sigma = sigma
            break

    # Re-create the reference using the selected fixed random seed.
    rng = np.random.default_rng(42)

    sampled = rng.choice(
        nasa_values,
        size=500,
        replace=True
    )

    reference = sampled * rng.lognormal(
        mean=0,
        sigma=selected_sigma,
        size=500
    )

    q1 = np.quantile(reference, 0.25)
    q3 = np.quantile(reference, 0.75)
    iqr = q3 - q1

    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr

    mean = np.mean(reference)
    std = np.std(reference)

    z_lower = mean - 3 * std
    z_upper = mean + 3 * std

    # Moderate anomalies are placed between the IQR and Z-score
    # upper limits. Extreme anomalies exceed both upper limits.
    if z_upper > iqr_upper:
        moderate_value = iqr_upper + 0.45 * (z_upper - iqr_upper)
    else:
        moderate_value = iqr_upper + 0.10

    extreme_value = max(z_upper, iqr_upper) + 0.30

    moderate_anomalies = np.repeat(moderate_value, 4)
    extreme_anomalies = np.repeat(extreme_value, 2)

    test_values = np.concatenate([
        reference,
        moderate_anomalies,
        extreme_anomalies
    ])

    # Ground truth: 0 = reference observation, 1 = injected anomaly
    ground_truth = np.concatenate([
        np.zeros(len(reference), dtype=int),
        np.ones(6, dtype=int)
    ])

    return {
        "reference": reference,
        "values": test_values,
        "ground_truth": ground_truth,
        "iqr_lower": iqr_lower,
        "iqr_upper": iqr_upper,
        "z_lower": z_lower,
        "z_upper": z_upper,
        "moderate_value": moderate_value,
        "extreme_value": extreme_value,
        "sigma": selected_sigma
    }


benchmark = create_skewed_benchmark(valid_capacity.to_numpy())

benchmark_values = benchmark["values"]
ground_truth = benchmark["ground_truth"]

iqr_predictions = (
    (benchmark_values < benchmark["iqr_lower"]) |
    (benchmark_values > benchmark["iqr_upper"])
).astype(int)

z_predictions = (
    (benchmark_values < benchmark["z_lower"]) |
    (benchmark_values > benchmark["z_upper"])
).astype(int)

iqr_scores = metric_scores(ground_truth, iqr_predictions)
z_scores = metric_scores(ground_truth, z_predictions)


# ------------------------------------------------------------
# Show detection limits
# ------------------------------------------------------------

st.markdown("### Detection limits")

limit_col1, limit_col2 = st.columns(2)

with limit_col1:
    st.markdown("#### IQR method")

    st.metric(
        "Lower fence",
        f'{benchmark["iqr_lower"]:.4f} Ah'
    )

    st.metric(
        "Upper fence",
        f'{benchmark["iqr_upper"]:.4f} Ah'
    )

    st.caption("Values outside Q1 − 1.5 × IQR and Q3 + 1.5 × IQR are flagged.")

with limit_col2:
    st.markdown("#### Z-score method")

    st.metric(
        "Lower limit",
        f'{benchmark["z_lower"]:.4f} Ah'
    )

    st.metric(
        "Upper limit",
        f'{benchmark["z_upper"]:.4f} Ah'
    )

    st.caption("Values outside the mean ± 3 standard deviations are flagged.")


# ------------------------------------------------------------
# Box plots
# ------------------------------------------------------------

st.markdown("### Visual comparison using box plots")

box_col1, box_col2 = st.columns(2)

with box_col1:
    st.plotly_chart(
        make_boxplot(
            pd.DataFrame({"capacity": benchmark_values}),
            "capacity",
            "IQR — detected observations",
            pd.Series(iqr_predictions.astype(bool))
        ),
        use_container_width=True
    )

with box_col2:
    st.plotly_chart(
        make_boxplot(
            pd.DataFrame({"capacity": benchmark_values}),
            "capacity",
            "Z-score — detected observations",
            pd.Series(z_predictions.astype(bool))
        ),
        use_container_width=True
    )


# ------------------------------------------------------------
# Detection results
# ------------------------------------------------------------

st.markdown("### Detection performance")

performance = pd.DataFrame(
    [iqr_scores, z_scores],
    index=["IQR", "Z-score"]
)

performance_display = performance.copy()

for col in ["Precision", "Recall", "F1-score"]:
    performance_display[col] = (
        performance_display[col] * 100
    ).round(2).astype(str) + "%"

st.dataframe(
    performance_display,
    use_container_width=True
)

p1, p2 = st.columns(2)

with p1:
    st.markdown("#### IQR detected values")

    iqr_detected = benchmark_values[iqr_predictions == 1]

    st.metric("Total detected", len(iqr_detected))

    st.dataframe(
        pd.DataFrame({
            "Detected capacity (Ah)": np.round(iqr_detected, 4)
        }),
        use_container_width=True,
        hide_index=True
    )

with p2:
    st.markdown("#### Z-score detected values")

    z_detected = benchmark_values[z_predictions == 1]

    st.metric("Total detected", len(z_detected))

    st.dataframe(
        pd.DataFrame({
            "Detected capacity (Ah)": np.round(z_detected, 4)
        }),
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------
# Explain the benchmark result
# ------------------------------------------------------------

st.markdown("### Experiment inference")

if iqr_scores["Recall"] > z_scores["Recall"]:
    st.success(
        "In this controlled right-skewed experiment, IQR identifies "
        "more of the labelled anomalies than Z-score."
    )
elif iqr_scores["Recall"] == z_scores["Recall"]:
    st.info(
        "Both methods identify the same proportion of labelled anomalies "
        "in this experiment."
    )
else:
    st.info(
        "Z-score identifies a greater proportion of labelled anomalies "
        "in this experiment."
    )

st.write(
    "The right-skewed distribution increases the influence of high values "
    "on the mean and standard deviation. This can widen the Z-score limits. "
    "IQR relies on quartiles and is less influenced by extreme values."
)

st.caption(
    "This is a controlled demonstration, not proof that IQR is universally "
    "more accurate. The best method depends on the distribution and the "
    "type of anomalies in the real dataset."
)

st.divider()


# ============================================================
# SECTION 4 — APPLY PREPROCESSING TO THE DEMONSTRATION DATA
# ============================================================

st.markdown('<div class="section-label">04 / CLEANING WORKSPACE</div>',
            unsafe_allow_html=True)

st.markdown("## Apply preprocessing to the dataset")

st.write(
    "Now apply the selected cleaning methods to the imperfect NASA "
    "demonstration copy. The selected method affects the final cleaned data."
)

working_df = dirty_df.copy()

# Handle missing values
if missing_method == "Median imputation":
    fill_value = working_df[CAPACITY_COL].median()
    working_df[CAPACITY_COL] = working_df[CAPACITY_COL].fillna(fill_value)

elif missing_method == "Mean imputation":
    fill_value = working_df[CAPACITY_COL].mean()
    working_df[CAPACITY_COL] = working_df[CAPACITY_COL].fillna(fill_value)

else:
    working_df = working_df.dropna(
        subset=[CAPACITY_COL]
    ).copy()


# Handle duplicates
if remove_duplicates:
    working_df = working_df.drop_duplicates(
        keep="first"
    ).copy()


# Outlier method selection
st.markdown("### Select outlier detection method")

outlier_method = st.radio(
    "Outlier detection",
    ["IQR", "Z-score"],
    horizontal=True,
    key="final_outlier_method"
)

outlier_treatment = st.selectbox(
    "Outlier treatment",
    [
        "Keep outliers",
        "Remove outliers",
        "Cap outliers"
    ],
    key="final_outlier_treatment"
)

if outlier_method == "IQR":
    out_lower, out_upper, q1, q3, iqr = calculate_iqr_limits(
        working_df[CAPACITY_COL]
    )

else:
    out_lower, out_upper, mean, std = calculate_zscore_limits(
        working_df[CAPACITY_COL]
    )

outlier_mask = get_outlier_mask(
    working_df[CAPACITY_COL],
    outlier_method
)

detected_count = int(outlier_mask.sum())

st.markdown("### Selected method limits")

o1, o2, o3 = st.columns(3)

o1.metric("Lower limit", f"{out_lower:.4f} Ah")
o2.metric("Upper limit", f"{out_upper:.4f} Ah")
o3.metric("Outliers detected", detected_count)

st.markdown("### Detected records")

if detected_count > 0:
    st.dataframe(
        working_df.loc[outlier_mask],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No outliers were detected using the selected method.")


# Before-treatment box plot
st.markdown("### Before outlier treatment")

st.plotly_chart(
    make_boxplot(
        working_df,
        CAPACITY_COL,
        f"Before treatment — {outlier_method}",
        outlier_mask
    ),
    use_container_width=True
)


# Apply outlier treatment
final_df = working_df.copy()

if outlier_treatment == "Remove outliers":
    final_df = final_df.loc[~outlier_mask].copy()

elif outlier_treatment == "Cap outliers":
    final_df.loc[
        final_df[CAPACITY_COL] < out_lower,
        CAPACITY_COL
    ] = out_lower

    final_df.loc[
        final_df[CAPACITY_COL] > out_upper,
        CAPACITY_COL
    ] = out_upper


# After-treatment box plot
st.markdown("### After outlier treatment")

st.plotly_chart(
    make_boxplot(
        final_df,
        CAPACITY_COL,
        f"After treatment — {outlier_treatment}"
    ),
    use_container_width=True
)


# ============================================================
# SECTION 5 — FINAL CLEANING SUMMARY
# ============================================================

st.markdown('<div class="section-label">05 / VERIFICATION</div>',
            unsafe_allow_html=True)

st.markdown("## Final preprocessing results")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

summary_col1.metric(
    "Records before",
    len(dirty_df)
)

summary_col2.metric(
    "Records after",
    len(final_df)
)

summary_col3.metric(
    "Missing values remaining",
    int(final_df[CAPACITY_COL].isna().sum())
)

summary_col4.metric(
    "Duplicate rows remaining",
    int(final_df.duplicated().sum())
)

summary_table = pd.DataFrame({
    "Preprocessing operation": [
        "Missing values",
        "Duplicate records",
        "Outlier detection",
        "Outlier treatment"
    ],
    "Selected method": [
        missing_method,
        "Remove duplicates" if remove_duplicates else "Keep duplicates",
        outlier_method,
        outlier_treatment
    ],
    "Result": [
        f"{int(dirty_df[CAPACITY_COL].isna().sum())} missing values handled",
        f"{duplicate_count if remove_duplicates else 0} duplicate rows removed",
        f"{detected_count} outliers detected",
        f"{outlier_treatment} applied"
    ]
})

st.dataframe(
    summary_table,
    use_container_width=True,
    hide_index=True
)

st.markdown("### Final cleaned dataset")

st.dataframe(
    final_df.head(30),
    use_container_width=True,
    hide_index=True
)

csv_data = final_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download cleaned dataset (CSV)",
    data=csv_data,
    file_name="aurora_cleaned_capacity.csv",
    mime="text/csv",
    use_container_width=True
)

st.success(
    "Preprocessing demonstration completed. "
    "The original NASA dataset has not been modified."
)