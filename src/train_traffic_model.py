import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

INPUT_PATH = "outputs/ml_traffic_data.csv"

print("AI Traffic Intelligence - ML Model Training")
print()
print("Loading prepared ML dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")

print()
print("================================")
print("        DATASET INFORMATION")
print("================================")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Features used by the model
FEATURES = [
    "vehicle_count",
    "average_speed_kmh",
    "congestion_score",
    "density_encoded"
]

TARGET = "future_congestion"

X = df[FEATURES]
y = df[TARGET]

print()
print("================================")
print("          ML FEATURES")
print("================================")

for feature in FEATURES:
    print(f"- {feature}")

print()
print(f"Target: {TARGET}")

print()
print("================================")
print("       TARGET DISTRIBUTION")
print("================================")

print(y.value_counts().sort_index())

# Time-based train/test split
# First 80% = training
# Last 20% = testing

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

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print()
print("Feature scaling completed!")

# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print()
print("Training Logistic Regression model...")

model.fit(X_train_scaled, y_train)

print("Model training completed!")

# Predictions
y_pred = model.predict(X_test_scaled)

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
print("      SAMPLE PREDICTIONS")
print("================================")

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(comparison.head(20).to_string(index=False))

print()
print("================================")
print("       MODEL TRAINING DONE")
print("================================")

print("Target:")
print("0 = NORMAL")
print("1 = MODERATE")

print()
print("Traffic prediction model completed!")

print("================================")