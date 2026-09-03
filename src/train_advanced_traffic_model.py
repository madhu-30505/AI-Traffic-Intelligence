import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_PATH = "outputs/traffic_ml_features.csv"


FEATURES = [
    "vehicle_count",
    "average_speed_kmh",
    "congestion_score",
    "congestion_target",

    "vehicle_count_lag_1",
    "vehicle_count_lag_3",
    "vehicle_count_lag_5",

    "speed_lag_1",
    "speed_lag_3",
    "speed_lag_5",

    "vehicle_count_rolling_mean",
    "speed_rolling_mean",
    "congestion_score_rolling_mean",

    "vehicle_count_change",
    "speed_change"
]


TARGET = "future_congestion"


# ==========================================
# START
# ==========================================

print("================================")
print(" AI Traffic Intelligence")
print(" Advanced Traffic Prediction")
print("================================")
print()


# ==========================================
# LOAD DATA
# ==========================================

print("Loading ML feature dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")

print()

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ==========================================
# PREPARE X AND Y
# ==========================================

X = df[FEATURES]

y = df[TARGET].astype(int)


print()

print("================================")
print("       DATASET INFORMATION")
print("================================")

print()

print("Features:")

for feature in FEATURES:
    print(f"- {feature}")

print()

print("Target:")
print("- future_congestion")


# ==========================================
# TARGET DISTRIBUTION
# ==========================================

print()

print("================================")
print("      TARGET DISTRIBUTION")
print("================================")

target_distribution = (
    y.value_counts()
    .sort_index()
)

print(
    target_distribution
)

print()

print("Class labels:")

print("0 = NORMAL")
print("1 = MODERATE")
print("2 = HEAVY")


# ==========================================
# TIME-BASED TRAIN / TEST SPLIT
# ==========================================

split_index = int(
    len(df) * 0.80
)


X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]


y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print()

print("================================")
print("       TRAIN / TEST SPLIT")
print("================================")

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


print()

print("Training class distribution:")

print(
    y_train.value_counts()
    .sort_index()
)


print()

print("Testing class distribution:")

print(
    y_test.value_counts()
    .sort_index()
)


# ==========================================
# FEATURE SCALING
# ==========================================

print()

print("Scaling features...")

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


print(
    "Feature scaling completed!"
)


# ==========================================
# TRAIN LOGISTIC REGRESSION
# ==========================================

print()

print("================================")
print("       MODEL TRAINING")
print("================================")

print()

print(
    "Training Logistic Regression..."
)


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)


model.fit(
    X_train_scaled,
    y_train
)


print(
    "Model training completed!"
)


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(
    X_test_scaled
)


# ==========================================
# EVALUATION
# ==========================================

print()

print("================================")
print("       MODEL EVALUATION")
print("================================")


accuracy = accuracy_score(
    y_test,
    y_pred
)


print()

print(
    f"Accuracy: {accuracy:.2f}"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print()

print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "NORMAL",
            "MODERATE",
            "HEAVY"
        ],
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print()

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )
)


# ==========================================
# SAMPLE PREDICTIONS
# ==========================================

print()

print("================================")
print("       SAMPLE PREDICTIONS")
print("================================")


comparison = pd.DataFrame({

    "Time": df.iloc[
        split_index:
    ][
        "video_time_seconds"
    ].values,

    "Actual": y_test.values,

    "Predicted": y_pred

})


print(
    comparison.head(30)
    .to_string(index=False)
)


# ==========================================
# MODEL COEFFICIENTS
# ==========================================

print()

print("================================")
print("       MODEL INFORMATION")
print("================================")

print()

print(
    "Model: Logistic Regression"
)

print(
    "Class weighting: balanced"
)

print(
    "Prediction horizon: 5 seconds"
)

print()

print(
    "The model uses current traffic,"
)

print(
    "previous traffic, and rolling"
)

print(
    "traffic trends to predict"
)

print(
    "future congestion."
)


# ==========================================
# COMPLETED
# ==========================================

print()

print("================================")
print("      MODEL COMPLETED")
print("================================")

print()

print(
    "Advanced 5-second traffic"
)

print(
    "prediction completed!"
)

print("================================")
