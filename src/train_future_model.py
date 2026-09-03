import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

INPUT_PATH = "outputs/future_traffic_data.csv"

print("AI Traffic Intelligence - 5 Second Future Prediction")
print()
print("Loading future prediction dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")

FEATURES = [
    "vehicle_count",
    "average_speed_kmh",
    "congestion_score",
    "congestion_target"
]

TARGET = "future_congestion"

X = df[FEATURES]
y = df[TARGET]

print()
print("================================")
print("          DATASET")
print("================================")

print(f"Rows: {len(df)}")
print(f"Features: {len(FEATURES)}")

print()
print("Features:")

for feature in FEATURES:
    print(f"- {feature}")

print()
print("Target:")
print("- future_congestion")

print()
print("================================")
print("      TARGET DISTRIBUTION")
print("================================")

print(
    y.value_counts().sort_index()
)

# Time-based split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print()
print("================================")
print("       TRAIN / TEST SPLIT")
print("================================")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print()
print("Feature scaling completed!")

# Train model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print()
print("Training Logistic Regression...")

model.fit(
    X_train_scaled,
    y_train
)

print("Model training completed!")

# Prediction
y_pred = model.predict(
    X_test_scaled
)

print()
print("================================")
print("        MODEL EVALUATION")
print("================================")

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"Accuracy: {accuracy:.2f}"
)

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "NORMAL",
            "MODERATE"
        ],
        zero_division=0
    )
)

print()
print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )
)

print()
print("================================")
print("       SAMPLE PREDICTIONS")
print("================================")

comparison = pd.DataFrame({
    "Time": df.iloc[
        split_index:
    ]["video_time_seconds"].values,

    "Actual": y_test.values,

    "Predicted": y_pred
})

print(
    comparison.to_string(
        index=False
    )
)

print()
print("================================")
print("      MODEL COMPLETED")
print("================================")

print(
    "Prediction horizon: 5 seconds"
)

print(
    "0 = NORMAL"
)

print(
    "1 = MODERATE"
)

print()
print(
    "5-second traffic prediction completed!"
)

print("================================")