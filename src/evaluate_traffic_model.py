import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)


INPUT_PATH = "outputs/traffic_1sec_ml_features.csv"
MODEL_PATH = "models/traffic_forecasting_model.pkl"
SCALER_PATH = "models/traffic_forecasting_scaler.pkl"


print("======================================")
print(" AI Traffic Intelligence")
print(" Model Evaluation & Selection")
print("======================================")
print()


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

features = [
    "vehicle_count",
    "average_speed_kmh",
    "congestion_score",

    "vehicle_count_lag_1",
    "vehicle_count_lag_2",
    "vehicle_count_lag_3",
    "vehicle_count_lag_5",

    "speed_lag_1",
    "speed_lag_2",
    "speed_lag_3",
    "speed_lag_5",

    "congestion_score_lag_1",
    "congestion_score_lag_3",
    "congestion_score_lag_5",

    "vehicle_count_rolling_3",
    "vehicle_count_rolling_5",

    "speed_rolling_3",
    "speed_rolling_5",

    "congestion_score_rolling_3",
    "congestion_score_rolling_5",

    "vehicle_count_change",
    "speed_change"
]

X = df[features]
y = df["future_congestion"]


# --------------------------------------------------
# 2. Time-based split
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print()


# --------------------------------------------------
# 3. Train Logistic Regression
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_predictions = (
    logistic_model.predict(X_test_scaled)
)


# --------------------------------------------------
# 4. Train Random Forest
# --------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = (
    rf_model.predict(X_test)
)


# --------------------------------------------------
# 5. Calculate metrics
# --------------------------------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)


logistic_balanced = balanced_accuracy_score(
    y_test,
    logistic_predictions
)

rf_balanced = balanced_accuracy_score(
    y_test,
    rf_predictions
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("======================================")
print(" LOGISTIC REGRESSION")
print("======================================")

print(
    f"Accuracy: {logistic_accuracy:.2f}"
)

print(
    f"Balanced Accuracy: {logistic_balanced:.2f}"
)

print()

print(
    classification_report(
        y_test,
        logistic_predictions,
        labels=[0, 1, 2],
        target_names=[
            "NORMAL",
            "MODERATE",
            "HEAVY"
        ],
        zero_division=0
    )
)


print("======================================")
print(" RANDOM FOREST")
print("======================================")

print(
    f"Accuracy: {rf_accuracy:.2f}"
)

print(
    f"Balanced Accuracy: {rf_balanced:.2f}"
)

print()

print(
    classification_report(
        y_test,
        rf_predictions,
        labels=[0, 1, 2],
        target_names=[
            "NORMAL",
            "MODERATE",
            "HEAVY"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# 7. Select model
# --------------------------------------------------

if rf_balanced >= logistic_balanced:

    best_model = rf_model
    best_model_name = "Random Forest"
    best_scaler = None

else:

    best_model = logistic_model
    best_model_name = "Logistic Regression"
    best_scaler = scaler


# --------------------------------------------------
# 8. Save selected model
# --------------------------------------------------

joblib.dump(
    best_model,
    MODEL_PATH
)

if best_scaler is not None:

    joblib.dump(
        best_scaler,
        SCALER_PATH
    )


# --------------------------------------------------
# 9. Save model metadata
# --------------------------------------------------

metadata = {
    "model_name": best_model_name,
    "features": features,
    "forecast_horizon_seconds": 5,
    "test_accuracy": (
        rf_accuracy
        if best_model_name == "Random Forest"
        else logistic_accuracy
    ),
    "test_balanced_accuracy": (
        rf_balanced
        if best_model_name == "Random Forest"
        else logistic_balanced
    )
}

metadata_df = pd.DataFrame(
    [metadata]
)

metadata_df.to_csv(
    "outputs/model_metadata.csv",
    index=False
)


# --------------------------------------------------
# 10. Final output
# --------------------------------------------------

print()
print("======================================")
print(" BEST MODEL SELECTED")
print("======================================")

print(
    f"Selected model: {best_model_name}"
)

print()

print(
    "The model has been saved successfully."
)

print(
    f"Model file: {MODEL_PATH}"
)

if best_scaler is not None:

    print(
        f"Scaler file: {SCALER_PATH}"
    )

print()
print(
    "IMPORTANT:"
)

print(
    "The current dataset is very small and"
)

print(
    "highly imbalanced."
)

print(
    "Therefore, evaluation results should be"
)

print(
    "treated as a pipeline demonstration."
)

print("======================================")
