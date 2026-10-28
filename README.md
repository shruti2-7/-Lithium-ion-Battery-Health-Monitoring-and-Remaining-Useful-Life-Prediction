# -Lithium-ion-Battery-Health-Monitoring-and-Remaining-Useful-Life-Prediction

## Topic 
**Lithium-ion Battery Health Monitoring and Remaining Useful Life Prediction Using Data Analytics**

Lithium-ion batteries power everything from phones and laptops to electric vehicles, but their performance quietly degrades with every charge-discharge cycle. Inside the cell, side reactions grow internal resistance and consume active material, so capacity fades gradually until the battery can no longer hold enough charge to be useful. Because this wear is invisible from the outside, users and systems cannot easily tell how much life is actually left. Data analytics offers a solution: the battery's own voltage, current and temperature measurements contain patterns that reveal its aging. By analyzing these measurements cycle by cycle, we can build a model that estimates the battery's health and predicts how many cycles remain before it reaches end of life.

The specific dataset behind this project is the NASA Prognostics Center of Excellence Battery Data Set, in which commercial 18650 lithium-ion cells were cycled through charge, discharge and impedance measurements at different temperatures until capacity faded from about 2 Ah toward roughly 1.4 Ah. The dataset is publicly available through NASA's PCoE repository.

## Problem Statement

Battery degradation is gradual and invisible, so users cannot know a battery's true remaining life from simple indicators like runtime alone. Overestimating life leads to unexpected failures and safety risks; replacing batteries too early wastes money and resources. This project addresses that gap with the following statement:

"To develop a data analytics pipeline that extracts health-indicative features from lithium-ion battery charge and discharge cycle data, and builds machine learning models that predict battery capacity fade and Remaining Useful Life (RUL), enabling condition-based battery replacement instead of fixed schedules."

In concrete terms: given measurements from the first N cycles of a cell, predict the capacity at future cycles and the number of cycles remaining before capacity drops below 80 percent of rated value.

## Data Collection

- **Source:** NASA PCoE Battery Data Set, free and publicly downloadable.
- **Contents:** Four cells (B0005, B0006, B0007, B0018). Each cell's record contains three repeating cycle types: charge cycles, discharge cycles and impedance measurement cycles.
- **Measurements per cycle:** time series of voltage, current and surface temperature, sampled at about 10 Hz, plus the measured discharge capacity that serves as the health label.
- **Format:** MATLAB.mat files with nested structures, one file per cell.
- **Tools:** Python, SciPy (loadmat), pandas.

## Data Preprocessing

The raw files are not analysis-ready, so this stage converts them into a clean structured dataset:

1. **Unwrapping:** Load each.mat file and flatten the nested MATLAB cycle structures into lists of Python dictionaries.
2. **Cycle separation:** Classify every cycle as charge, discharge or impedance, and keep discharge cycles because they carry the health information (capacity).
3. **Extraction:** For every discharge cycle, extract the discharge capacity, discharge duration, ambient temperature and the voltage, current and temperature time series.
4. **Cleaning:** Remove incomplete or anomalous cycles, verify capacities decrease (mostly) monotonically with cycle number, and handle the occasional capacity rebound after rest periods.
5. **Structuring:** Assemble one master table with one row per cycle: cell ID, cycle number, capacity, duration, temperature, and references to the raw curves.

Result: a clean DataFrame of roughly 500 to 700 cycles across the four cells, each labeled with its true capacity.

## Data Transformations and Feature Engineering

The model cannot read raw time series, so each cycle is transformed into numerical features that physically reflect aging:

- **Voltage slope during discharge** (mV per second): steepens as the cell weakens.
- **CC charge time and CV charge time** and their ratio: the constant-current/constant-voltage split shifts with aging.
- **Initial voltage drop** just after discharge starts: rises with internal resistance growth.
- **Average and maximum surface temperature** during discharge.
- **Discharge duration and average discharge voltage.**

Supporting transformations include smoothing (rolling averages) to reduce measurement noise, normalization/standardization of features for model stability, and computing the target label RUL as (end-of-life cycle number minus current cycle number).

## ML Visualization (EDA)

Exploratory visualizations both explain the data and feed the poster:

- **Capacity fade curves:** capacity vs cycle number per cell, with the 80 percent end-of-life threshold marked. This is the signature plot of the project.
- **Discharge curve evolution:** voltage-vs-time curves overlaid for early, middle and late cycles, showing the shape change with aging.
- **Temperature effect:** comparing cells cycled at different temperatures to show accelerated aging.
- **Correlation heatmap:** which engineered features correlate most with capacity, guiding model inputs.
- **Feature trend plots:** each candidate feature plotted against cycle number to confirm it moves monotonically with aging.

These plots are built with Matplotlib and Seaborn, and later become the Streamlit dashboard's EDA page.

## ML Modeling

Framed as supervised regression: given features of cycle N, predict capacity (from which RUL follows). Three models, in increasing complexity:

1. **Linear Regression (baseline):** fits the near-linear early capacity fade; simple and interpretable.
2. **Random Forest Regressor:** tree ensemble capturing nonlinear degradation behavior, robust on small datasets.
3. **Gradient Boosting Regressor:** usually the most accurate on tabular degradation features.

Data split is done the honest way: train on a subset of cells, test on a held-out cell the model has never seen, since splitting random cycles within a cell leaks information. Hyperparameters are tuned by grid search with cross-validation inside the training set only.

## Results and Interpretation

- **Metrics:** RMSE and MAE between predicted and measured capacity (and RUL) on the test cell; the expected outcome is that Gradient Boosting beats Random Forest beats linear baseline.
- **Key figures:** predicted vs actual degradation curve overlaid for the test cell, error-vs-cycle-position plot showing where predictions are weakest (typically near the end-of-life knee), and feature importance from the tree models, which usually highlights voltage slope and charge-time features as the best aging indicators.
- **Interpretation:** the model confirms that simple electrical measurements encode battery health, meaning a cheap monitoring scheme can replace expensive periodic capacity tests. Practically, this supports condition-based maintenance: replace or service a battery when predicted RUL runs low, not on a fixed calendar.
- **Deliverables:** the analysis notebook, the Streamlit interactive dashboard, the Power BI results dashboard, and the workflow poster.

That is the full narrative from topic to interpretation. When you are ready, tell me and we will design the poster layout around exactly these sections, and then start building step 1 of the actual pipeline.
