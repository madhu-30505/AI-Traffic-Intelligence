import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

INPUT_PATH = "outputs/ml_traffic_data.csv"

print("AI Traffic Intelligence - Random Forest Training")
print()
print("Loading prepared ML dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")

FEATURES = [
    "vehicle_count",
    "average_speed_kmh",
    "congestion_score",
    "density_encoded"
]

TARGET = "future_congestion"

X = df[FEATURES]
y = df[TARGET]

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

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

print()
print("Training Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")

# Predictions
y_pred = model.predict(X_test)

print()
print("================================")
print("        MODEL EVALUATION")
print("================================")

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["NORMAL", "MODERATE"],
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
print("      FEATURE IMPORTANCE")
print("================================")

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))

print()
print("================================")
print("    RANDOM FOREST COMPLETED")
print("================================")