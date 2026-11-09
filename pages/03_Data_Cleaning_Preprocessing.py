import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =========================================================
# 1. PAGE SETUP AND DATA LOADING
# =========================================================

st.title("Data Cleaning & Preprocessing")

st.write(
    "Assess and prepare the actual NASA lithium-ion battery "
    "dataset before feature engineering and predictive modelling."
)

st.divider()

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "discharge_capacity.csv"
)

if not DATA_PATH.exists():
    st.error("NASA processed dataset could not be found.")
    st.caption(f"Expected location: {DATA_PATH}")
    st.stop()

original_df = pd.read_csv(DATA_PATH)
df = original_df.copy()

required_columns = [
    "battery",
    "record_idx",
    "discharge_idx",
    "capacity_Ah",
    "ambient_temperature_C"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Required columns missing: {missing_columns}")
    st.stop()

# =========================================================
# 2. DATASET QUALITY ASSESSMENT
# =========================================================

st.header("01 · Data Quality Assessment")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Features", len(df.columns))

with col3:
    st.metric("Battery Cells", df["battery"].nunique())

with col4:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

st.markdown("**Dataset Preview**")

st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True
)

st.markdown("**Records by Battery Cell**")

battery_counts = (
    df.groupby("battery")
    .size()
    .reset_index(name="Discharge Records")
)

st.dataframe(
    battery_counts,
    use_container_width=True,
    hide_index=True
)

with st.expander("View descriptive statistics"):
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

st.divider()

# =========================================================
# 3. MISSING VALUE ANALYSIS
# =========================================================

st.header("02 · Missing Value Analysis")

missing_counts = df.isnull().sum()

missing_report = pd.DataFrame({
    "Feature": missing_counts.index,
    "Missing Values": missing_counts.values,
    "Missing Percentage": (
        missing_counts.values / len(df) * 100
    ).round(2)
})

total_missing = int(missing_counts.sum())

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Missing Values", total_missing)

with col2:
    st.metric(
        "Features Affected",
        int((missing_counts > 0).sum())
    )

st.dataframe(
    missing_report,
    use_container_width=True,
    hide_index=True
)

if total_missing == 0:
    st.success(
        "No missing values were found in the actual processed dataset."
    )
else:
    st.warning(
        "Missing values were detected. Their handling is configured "
        "in the preprocessing section below."
    )

st.divider()

# =========================================================
# 4. DUPLICATE RECORD DETECTION
# =========================================================

st.header("03 · Duplicate Record Detection")

exact_duplicates = int(df.duplicated().sum())

cycle_duplicates = int(
    df.duplicated(
        subset=["battery", "discharge_idx"]
    ).sum()
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Exact Duplicate Rows", exact_duplicates)

with col2:
    st.metric("Repeated Battery-Cycle Entries", cycle_duplicates)

st.caption(
    "Exact duplicates have identical values across all columns. "
    "Repeated battery-cycle entries share the same battery ID "
    "and discharge-cycle index."
)

if exact_duplicates == 0 and cycle_duplicates == 0:
    st.success("No duplicate records were detected.")
else:
    st.warning("Duplicate records require removal during cleaning.")

st.divider()

# =========================================================
# 5. FEATURE TYPE VALIDATION
# =========================================================

st.header("04 · Feature Type Validation")

expected_types = {
    "battery": "Categorical",
    "record_idx": "Integer",
    "discharge_idx": "Integer",
    "capacity_Ah": "Numeric",
    "ambient_temperature_C": "Numeric"
}

type_rows = []

for column in df.columns:

    expected = expected_types.get(column, "Review required")
    actual = str(df[column].dtype)

    if expected == "Categorical":
        status = (
            "Valid"
            if pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            else "Review"
        )

    elif expected == "Integer":
        status = (
            "Valid"
            if pd.api.types.is_integer_dtype(df[column])
            else "Review"
        )

    elif expected == "Numeric":
        status = (
            "Valid"
            if pd.api.types.is_numeric_dtype(df[column])
            else "Review"
        )

    else:
        status = "Review"

    type_rows.append({
        "Feature": column,
        "Expected Type": expected,
        "Actual Data Type": actual,
        "Validation": status
    })

type_report = pd.DataFrame(type_rows)

st.dataframe(
    type_report,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =========================================================
# 6. CAPACITY OUTLIER INSPECTION
# =========================================================

st.header("05 · Capacity Outlier Inspection")

st.write(
    "Capacity measurements are inspected using the Interquartile "
    "Range (IQR) method, separately for each battery cell."
)

outlier_frames = []

for battery in df["battery"].dropna().unique():

    battery_data = df[
        df["battery"] == battery
    ].copy()

    q1 = battery_data["capacity_Ah"].quantile(0.25)
    q3 = battery_data["capacity_Ah"].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    battery_data["IQR Status"] = "Within Range"

    mask = (
        (battery_data["capacity_Ah"] < lower_limit)
        | (battery_data["capacity_Ah"] > upper_limit)
    )

    battery_data.loc[mask, "IQR Status"] = "Flagged"

    outlier_frames.append({
        "Battery": battery,
        "Q1": round(q1, 4),
        "Q3": round(q3, 4),
        "IQR": round(iqr, 4),
        "Lower Limit": round(lower_limit, 4),
        "Upper Limit": round(upper_limit, 4),
        "Flagged Measurements": int(mask.sum())
    })

outlier_report = pd.DataFrame(outlier_frames)

st.dataframe(
    outlier_report,
    use_container_width=True,
    hide_index=True
)

fig = px.box(
    df,
    x="battery",
    y="capacity_Ah",
    color="battery",
    points="outliers",
    title="Discharge Capacity Distribution by Battery"
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    xaxis_title="Battery Cell",
    yaxis_title="Discharge Capacity (Ah)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info(
    "IQR identifies unusual capacity measurements for inspection. "
    "They are retained because battery degradation and capacity "
    "rebound can produce genuine fluctuations."
)

st.divider()

# =========================================================
# 7. DATA PREPROCESSING
# =========================================================

st.header("06 · Data Preprocessing")

st.write(
    "Apply cleaning operations to a separate working copy. "
    "The original processed NASA dataset remains unchanged."
)

missing_strategy = st.selectbox(
    "Missing Value Handling Strategy",
    [
        "Median Imputation",
        "Mean Imputation",
        "Drop Rows with Missing Values"
    ]
)

if st.button("Apply Data Cleaning", type="primary"):

    cleaned_df = original_df.copy()

    initial_rows = len(cleaned_df)

    # Remove exact duplicate rows
    exact_removed = int(cleaned_df.duplicated().sum())

    cleaned_df = cleaned_df.drop_duplicates()

    # Remove repeated battery-cycle entries
    cycle_removed = int(
        cleaned_df.duplicated(
            subset=["battery", "discharge_idx"]
        ).sum()
    )

    cleaned_df = cleaned_df.drop_duplicates(
        subset=["battery", "discharge_idx"],
        keep="first"
    )

    # Handle missing values
    if missing_strategy == "Drop Rows with Missing Values":

        cleaned_df = cleaned_df.dropna()

    else:

        measurement_columns = [
            col for col in [
                "capacity_Ah",
                "ambient_temperature_C"
            ]
            if col in cleaned_df.columns
        ]

        for column in measurement_columns:

            if cleaned_df[column].isnull().any():

                if missing_strategy == "Median Imputation":
                    fill_value = cleaned_df[column].median()

                else:
                    fill_value = cleaned_df[column].mean()

                if pd.notna(fill_value):
                    cleaned_df[column] = cleaned_df[column].fillna(
                        fill_value
                    )

    st.session_state["aurora_cleaned_df"] = cleaned_df
    st.session_state["aurora_cleaning_report"] = {
        "Initial Rows": initial_rows,
        "Final Rows": len(cleaned_df),
        "Exact Duplicates Removed": exact_removed,
        "Repeated Battery-Cycle Entries Removed": cycle_removed,
        "Missing Values Remaining": int(
            cleaned_df.isnull().sum().sum()
        ),
        "Strategy": missing_strategy
    }

    st.success("Data cleaning completed.")

# =========================================================
# 8. BEFORE AND AFTER COMPARISON
# =========================================================

st.header("07 · Cleaning Results & Final Dataset")

if "aurora_cleaned_df" in st.session_state:

    cleaned_df = st.session_state["aurora_cleaned_df"]
    report = st.session_state["aurora_cleaning_report"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Records",
            report["Initial Rows"]
        )

    with col2:
        st.metric(
            "Cleaned Records",
            report["Final Rows"]
        )

    with col3:
        st.metric(
            "Records Removed",
            report["Initial Rows"] - report["Final Rows"]
        )

    comparison = pd.DataFrame({
        "Data Quality Measure": [
            "Total Records",
            "Missing Values",
            "Exact Duplicate Rows",
            "Repeated Battery-Cycle Entries"
        ],
        "Before Cleaning": [
            len(original_df),
            int(original_df.isnull().sum().sum()),
            int(original_df.duplicated().sum()),
            int(original_df.duplicated(
                subset=["battery", "discharge_idx"]
            ).sum())
        ],
        "After Cleaning": [
            len(cleaned_df),
            int(cleaned_df.isnull().sum().sum()),
            int(cleaned_df.duplicated().sum()),
            int(cleaned_df.duplicated(
                subset=["battery", "discharge_idx"]
            ).sum())
        ]
    })

    st.markdown("**Before vs After Comparison**")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("**Cleaning Operations Applied**")

    st.write(f"- Missing-value strategy: {report['Strategy']}")
    st.write(
        f"- Exact duplicate rows removed: "
        f"{report['Exact Duplicates Removed']}"
    )
    st.write(
        f"- Repeated battery-cycle entries removed: "
        f"{report['Repeated Battery-Cycle Entries Removed']}"
    )
    st.write(
        f"- Missing values remaining: "
        f"{report['Missing Values Remaining']}"
    )

    st.markdown("**Final Cleaned Dataset Preview**")

    st.dataframe(
        cleaned_df.head(15),
        use_container_width=True,
        hide_index=True
    )

    csv_data = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Cleaned Dataset",
        data=csv_data,
        file_name="aurora_cleaned_discharge_capacity.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "The original processed NASA CSV is not overwritten. "
        "The cleaned dataset is available as a separate download."
    )

else:

    st.info(
        "Select a missing-value strategy and click "
        "'Apply Data Cleaning' to generate the cleaned dataset "
        "and before-and-after comparison."
    )