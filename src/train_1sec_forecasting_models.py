import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


INPUT_PATH = "outputs/traffic_1sec_ml_features.csv"


print("======================================")
print(" AI Traffic Intelligence")
print(" 1-Second Traffic Forecasting")
print("======================================")
print()


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("Loading ML dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Dataset rows: {len(df)}")
print()


# --------------------------------------------------
# 2. Define features
# --------------------------------------------------

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
# 3. Display target distribution
# --------------------------------------------------

print("Target distribution:")

target_distribution = (
    y.value_counts()
    .sort_index()
)

for label, count in target_distribution.items():

    if label == 0:
        name = "NORMAL"
    elif label == 1:
        name = "MODERATE"
    else:
        name = "HEAVY"

    print(f"{name}: {count}")

print()


# --------------------------------------------------
# 4. Time-based train/test split
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Time-based split:")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")
print()


# --------------------------------------------------
# 5. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 6. Logistic Regression
# --------------------------------------------------

print("======================================")
print(" MODEL 1: LOGISTIC REGRESSION")
print("======================================")

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

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print(
    f"Accuracy: {logistic_accuracy:.2f}"
)

print()

print("Classification Report:")

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

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        logistic_predictions,
        labels=[0, 1, 2]
    )
)

print()


# --------------------------------------------------
# 7. Random Forest
# --------------------------------------------------

print("======================================")
print(" MODEL 2: RANDOM FOREST")
print("======================================")

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

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(
    f"Accuracy: {rf_accuracy:.2f}"
)

print()

print("Classification Report:")

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

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_predictions,
        labels=[0, 1, 2]
    )
)

print()


# --------------------------------------------------
# 8. Gradient Boosting
# --------------------------------------------------

print("======================================")
print(" MODEL 3: GRADIENT BOOSTING")
print("======================================")

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)

gb_model.fit(
    X_train,
    y_train
)

gb_predictions = (
    gb_model.predict(X_test)
)

gb_accuracy = accuracy_score(
    y_test,
    gb_predictions
)

print(
    f"Accuracy: {gb_accuracy:.2f}"
)

print()

print("Classification Report:")

print(
    classification_report(
        y_test,
        gb_predictions,
        labels=[0, 1, 2],
        target_names=[
            "NORMAL",
            "MODERATE",
            "HEAVY"
        ],
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        gb_predictions,
        labels=[0, 1, 2]
    )
)

print()


# --------------------------------------------------
# 9. Compare models
# --------------------------------------------------

print("======================================")
print(" MODEL COMPARISON")
print("======================================")

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],

    "Accuracy": [
        logistic_accuracy,
        rf_accuracy,
        gb_accuracy
    ]
})

results = results.sort_values(
    "Accuracy",
    ascending=False
)

print(
    results.to_string(index=False)
)

print()


# --------------------------------------------------
# 10. Random Forest feature importance
# --------------------------------------------------

print("======================================")
print(" RANDOM FOREST FEATURE IMPORTANCE")
print("======================================")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(
    importance.head(10).to_string(index=False)
)

print()


# --------------------------------------------------
# 11. Final message
# --------------------------------------------------

print("======================================")
print(" FORECASTING TRAINING COMPLETED")
print("======================================")

print()
print(
    "Models trained successfully."
)

print(
    "Remember: this 39-second dataset is too small"
)

print(
    "for reliable real-world forecasting."
)

print(
    "Use the results as a pipeline demonstration,"
)

print(
    "not as production-level model performance."
)

print("======================================")
