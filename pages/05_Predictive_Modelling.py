
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# AURORA — PREDICTIVE MODELLING
# =========================================================

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
    st.error(f"Unable to load dataset: {error}")
    st.stop()


# =========================================================
# CREATE HISTORICAL PREDICTION FEATURES
# =========================================================

def create_prediction_features(data):

    df = data.copy()

    grouped = df.groupby("battery", sort=False)

    # Previous cycle capacity
    df["previous_capacity_Ah"] = (
        grouped["capacity_Ah"].shift(1)
    )

    # Capacity change from the previous cycle
    df["capacity_change_Ah"] = (
        df["capacity_Ah"] -
        df["previous_capacity_Ah"]
    )

    # Historical capacity change
    df["previous_capacity_change_Ah"] = (
        df.groupby("battery")["capacity_change_Ah"]
        .shift(1)
    )

    # Historical fade rate
    df["previous_fade_rate_Ah_per_cycle"] = (
        -df["previous_capacity_change_Ah"]
    )

    # Previous five-cycle mean
    df["rolling_mean_previous_5_Ah"] = (
        df.groupby("battery")["capacity_Ah"]
        .transform(
            lambda s: s.shift(1)
            .rolling(5, min_periods=5)
            .mean()
        )
    )

    # Previous five-cycle standard deviation
    df["rolling_std_previous_5_Ah"] = (
        df.groupby("battery")["capacity_Ah"]
        .transform(
            lambda s: s.shift(1)
            .rolling(5, min_periods=5)
            .std(ddof=1)
        )
    )

    return df


model_df = create_prediction_features(original_df)


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURES = [
    "previous_capacity_Ah",
    "previous_capacity_change_Ah",
    "previous_fade_rate_Ah_per_cycle",
    "rolling_mean_previous_5_Ah",
    "rolling_std_previous_5_Ah",
    "ambient_temperature_C",
    "discharge_idx"
]

TARGET = "capacity_Ah"


# =========================================================
# TRAINING AND TESTING DATA
# =========================================================

train_df = model_df[
    model_df["battery"].isin(TRAIN_BATTERIES)
].copy()

test_df = model_df[
    model_df["battery"] == TEST_BATTERY
].copy()

train_df = train_df.replace(
    [np.inf, -np.inf],
    np.nan
)

test_df = test_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove initial rows without enough historical measurements
train_df = train_df.dropna(
    subset=[
        "previous_capacity_Ah",
        "previous_capacity_change_Ah",
        "rolling_mean_previous_5_Ah",
        "rolling_std_previous_5_Ah"
    ]
).reset_index(drop=True)

test_df = test_df.dropna(
    subset=[
        "previous_capacity_Ah",
        "previous_capacity_change_Ah",
        "rolling_mean_previous_5_Ah",
        "rolling_std_previous_5_Ah"
    ]
).reset_index(drop=True)

if train_df.empty or test_df.empty:
    st.error(
        "Insufficient training or testing records after creating "
        "historical features."
    )
    st.stop()


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# =========================================================
# DEFINE MODELS
# =========================================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.04,
        max_depth=2,
        loss="huber",
        random_state=42
    )
}


# =========================================================
# TRAIN MODELS AND GENERATE PREDICTIONS
# =========================================================

results = {}
predictions = {}
trained_models = {}

for name, estimator in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    predicted_capacity = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predicted_capacity
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predicted_capacity
        )
    )

    r2 = r2_score(
        y_test,
        predicted_capacity
    )

    results[name] = {
        "MAE (Ah)": mae,
        "RMSE (Ah)": rmse,
        "R² Score": r2
    }

    predictions[name] = predicted_capacity

    trained_models[name] = pipeline


results_df = (
    pd.DataFrame(results)
    .T
    .sort_values("RMSE (Ah)")
)

best_model_name = results_df.index[0]
best_model = trained_models[best_model_name]

best_predictions = predictions[best_model_name]


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Predictive Modelling")
st.caption("AURORA  ·  BATTERY ANALYTICS")

st.markdown(
    "Machine learning models are trained on historical battery "
    "measurements to predict discharge capacity and estimate "
    "Remaining Useful Life."
)

st.divider()


# =========================================================
# 01 — MODELLING APPROACH
# =========================================================

st.header("01 — Modelling Approach")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Training dataset**")

    st.success("B0005 · B0006 · B0007")

    st.metric(
        "Training records",
        f"{len(train_df):,}"
    )

with col2:
    st.markdown("**Independent testing dataset**")

    st.info("B0018")

    st.metric(
        "Testing records",
        f"{len(test_df):,}"
    )

st.caption(
    "The complete B0018 cell is held out during training. "
    "Its previous-cycle measurements are used to predict subsequent "
    "cycles during testing."
)

st.subheader("Selected prediction inputs")

st.dataframe(
    pd.DataFrame({
        "Input feature": FEATURES,
        "Purpose": [
            "Capacity of the previous cycle",
            "Previous capacity change",
            "Historical capacity degradation",
            "Average capacity over previous five cycles",
            "Variation over previous five cycles",
            "Operating temperature",
            "Current discharge-cycle index"
        ]
    }),
    use_container_width=True,
    hide_index=True,
    height=290
)

st.markdown("**Prediction target:** Current discharge capacity (Ah)")


# =========================================================
# 02 — MODEL TRAINING
# =========================================================

st.divider()
st.header("02 — Model Training")

st.write(
    "Three regression algorithms are trained using the same "
    "historical input features and training batteries."
)

model_descriptions = {
    "Linear Regression": (
        "Fits a linear relationship between historical features "
        "and discharge capacity."
    ),
    "Random Forest": (
        "Combines multiple decision trees to capture nonlinear "
        "relationships in battery degradation."
    ),
    "Gradient Boosting": (
        "Builds decision trees sequentially, with each new tree "
        "correcting errors made by previous trees."
    )
}

for name, description in model_descriptions.items():

    with st.container(border=True):

        st.markdown(f"**{name}**")
        st.caption(description)

st.success(
    f"All three models have been trained and evaluated. "
    f"Best-performing model by RMSE: {best_model_name}."
)


# =========================================================
# 03 — MODEL EVALUATION
# =========================================================

st.divider()
st.header("03 — Model Evaluation")

st.write(
    "Model performance is evaluated using the independent B0018 "
    "test battery."
)

st.subheader("Evaluation metrics")

st.dataframe(
    results_df.round(5),
    use_container_width=True
)

best_metrics = results_df.loc[best_model_name]

c1, c2, c3 = st.columns(3)

c1.metric(
    "Best model",
    best_model_name
)

c2.metric(
    "MAE",
    f"{best_metrics['MAE (Ah)']:.4f} Ah"
)

c3.metric(
    "RMSE",
    f"{best_metrics['RMSE (Ah)']:.4f} Ah"
)

with st.expander("Understanding the evaluation metrics"):

    st.markdown("""
    **Mean Absolute Error (MAE)**

    Average absolute difference between actual and predicted capacity.
    Lower values indicate better predictions.

    **Root Mean Squared Error (RMSE)**

    Square root of the average squared prediction error.
    It gives greater weight to larger errors.

    **R² Score**

    Measures how well the model explains variation in actual capacity.
    A value closer to 1 indicates a better fit. It can be negative
    when predictions perform worse than a constant-mean baseline.
    """)


# Model comparison graph
st.subheader("Model performance comparison")

comparison_df = results_df.reset_index()
comparison_df = comparison_df.rename(
    columns={"index": "Model"}
)

fig_metrics = px.bar(
    comparison_df,
    x="Model",
    y="RMSE (Ah)",
    color="Model",
    text_auto=".4f",
    title="RMSE comparison — lower is better"
)

fig_metrics.update_layout(
    showlegend=False,
    height=350
)

st.plotly_chart(
    fig_metrics,
    use_container_width=True
)


# =========================================================
# 04 — ACTUAL VS PREDICTED CAPACITY
# =========================================================

st.divider()
st.header("04 — Actual vs Predicted Capacity")

st.write(
    "The graph compares the measured discharge capacity of B0018 "
    "with the predictions generated by the selected model."
)

plot_df = test_df[
    ["discharge_idx", "capacity_Ah"]
].copy()

plot_df["Predicted Capacity (Ah)"] = best_predictions

plot_df = plot_df.rename(
    columns={
        "capacity_Ah": "Actual Capacity (Ah)"
    }
)

fig_prediction = go.Figure()

fig_prediction.add_trace(
    go.Scatter(
        x=plot_df["discharge_idx"],
        y=plot_df["Actual Capacity (Ah)"],
        mode="lines",
        name="Actual Capacity"
    )
)

fig_prediction.add_trace(
    go.Scatter(
        x=plot_df["discharge_idx"],
        y=plot_df["Predicted Capacity (Ah)"],
        mode="lines",
        name="Predicted Capacity"
    )
)

fig_prediction.add_hline(
    y=EOL_CAPACITY,
    line_dash="dash",
    annotation_text="EOL threshold: 1.5 Ah"
)

fig_prediction.update_layout(
    title=f"{best_model_name} — B0018",
    xaxis_title="Discharge Cycle",
    yaxis_title="Capacity (Ah)",
    height=430,
    hovermode="x unified"
)

st.plotly_chart(
    fig_prediction,
    use_container_width=True
)


# Prediction error
plot_df["Prediction Error (Ah)"] = (
    plot_df["Actual Capacity (Ah)"] -
    plot_df["Predicted Capacity (Ah)"]
)

fig_error = px.line(
    plot_df,
    x="discharge_idx",
    y="Prediction Error (Ah)",
    title="Prediction error across discharge cycles"
)

fig_error.add_hline(y=0, line_dash="dash")

fig_error.update_layout(
    xaxis_title="Discharge Cycle",
    yaxis_title="Actual − Predicted (Ah)",
    height=320
)

st.plotly_chart(
    fig_error,
    use_container_width=True
)


# =========================================================
# 05 — FEATURE IMPORTANCE
# =========================================================

st.divider()
st.header("05 — Feature Importance")

st.write(
    "Feature importance shows how much each input contributes "
    "to the Random Forest model's predictions."
)

rf_pipeline = trained_models["Random Forest"]

rf_model = rf_pipeline.named_steps["model"]

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf_model.feature_importances_
}).sort_values(
    "Importance",
    ascending=True
)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Random Forest feature importance"
)

fig_importance.update_layout(
    height=370
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

st.caption(
    "Feature importance indicates the model's reliance on each feature. "
    "It does not establish causation."
)


# =========================================================
# 06 — REMAINING USEFUL LIFE
# =========================================================

st.divider()
st.header("06 — Remaining Useful Life (RUL)")

st.write(
    "RUL estimates how many additional discharge cycles remain "
    "before battery capacity reaches the 1.5 Ah end-of-life threshold."
)

st.latex(
    r"RUL=N_{\mathrm{EOL}}-N_{\mathrm{current}}"
)

st.markdown("""
- \(N_{\mathrm{EOL}}\): Estimated cycle at which capacity reaches 1.5 Ah.
- \(N_{\mathrm{current}}\): Latest observed discharge-cycle index.
""")

# Use the latest predicted point and recent predicted degradation trend
rul_window = min(20, len(plot_df))

recent_points = plot_df.tail(rul_window)

current_cycle = int(
    plot_df["discharge_idx"].iloc[-1]
)

current_predicted_capacity = float(
    plot_df["Predicted Capacity (Ah)"].iloc[-1]
)

current_actual_capacity = float(
    plot_df["Actual Capacity (Ah)"].iloc[-1]
)

# Fit a local linear trend to recent predictions
if len(recent_points) >= 3:

    trend_model = LinearRegression()

    trend_model.fit(
        recent_points[["discharge_idx"]],
        recent_points["Predicted Capacity (Ah)"]
    )

    degradation_slope = float(
        trend_model.coef_[0]
    )

else:
    degradation_slope = np.nan


rul_estimate = None
rul_status = ""

if current_predicted_capacity <= EOL_CAPACITY:

    rul_estimate = 0
    rul_status = (
        "The latest predicted capacity is already at or below "
        "the end-of-life threshold."
    )

elif np.isfinite(degradation_slope) and degradation_slope < 0:

    cycles_remaining = (
        (EOL_CAPACITY - current_predicted_capacity) /
        degradation_slope
    )

    rul_estimate = max(
        0,
        int(np.ceil(cycles_remaining))
    )

    rul_status = (
        "Estimated using the recent linear trend in predicted capacity."
    )

else:

    rul_status = (
        "A reliable positive RUL estimate cannot be calculated because "
        "the recent predicted capacity trend is not decreasing."
    )


rul_col1, rul_col2, rul_col3 = st.columns(3)

rul_col1.metric(
    "Latest actual capacity",
    f"{current_actual_capacity:.4f} Ah"
)

rul_col2.metric(
    "Latest predicted capacity",
    f"{current_predicted_capacity:.4f} Ah"
)

if rul_estimate is not None:

    rul_col3.metric(
        "Estimated RUL",
        f"{rul_estimate} cycles"
    )

else:

    rul_col3.metric(
        "Estimated RUL",
        "Not reliable"
    )

st.caption(rul_status)

st.warning(
    "RUL is an approximate extrapolation from the recent predicted "
    "capacity trend, not a separately trained RUL regression model. "
    "Its accuracy depends on the assumption that recent degradation "
    "continues at a similar rate."
)


# =========================================================
# 07 — PREDICTION RESULTS
# =========================================================

st.divider()
st.header("07 — Prediction Results")

results_output = plot_df[
    [
        "discharge_idx",
        "Actual Capacity (Ah)",
        "Predicted Capacity (Ah)",
        "Prediction Error (Ah)"
    ]
].copy()

st.dataframe(
    results_output.head(12).round(5),
    use_container_width=True,
    hide_index=True,
    height=300
)

st.download_button(
    "Download prediction results",
    data=results_output.to_csv(index=False).encode("utf-8"),
    file_name="aurora_prediction_results.csv",
    mime="text/csv"
)


# =========================================================
# SUMMARY
# =========================================================

st.divider()

st.subheader("Predictive Modelling Summary")

st.markdown(f"""
- Training completed using B0005, B0006 and B0007.
- Independent testing completed using B0018.
- Compared Linear Regression, Random Forest and Gradient Boosting.
- Best model by RMSE: **{best_model_name}**.
- Evaluated performance using MAE, RMSE and R².
- Generated actual-versus-predicted capacity comparisons.
- Estimated RUL using the 1.5 Ah end-of-life threshold.
""")

st.info(
    "The reported metrics and RUL are calculated from the actual "
    "NASA dataset. They should be interpreted as model estimates, "
    "not guaranteed battery-life values."
)
