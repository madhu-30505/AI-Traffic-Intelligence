import pandas as pd
import os

INPUT_PATH = "outputs/traffic_history.csv"
OUTPUT_PATH = "outputs/ml_traffic_data.csv"

print("AI Traffic Intelligence - ML Data Preparation")
print()
print("Loading traffic dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")

print()
print("================================")
print("       ORIGINAL DATA")
print("================================")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Create video-relative time
df["video_time_seconds"] = (
    df["frame_number"] - df["frame_number"].min()
) / 30.0

# Create time-based features
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["second"] = df["timestamp"].dt.second

# Convert categorical density into numerical values
density_mapping = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
}

df["density_encoded"] = df["density"].map(density_mapping)

# Convert congestion level into numerical target
congestion_mapping = {
    "NORMAL": 0,
    "MODERATE": 1,
    "HEAVY": 2
}

df["congestion_target"] = df["congestion_level"].map(
    congestion_mapping
)

# Create future congestion target
# The model will learn to predict the next traffic state.
df["future_congestion"] = df["congestion_target"].shift(-1)

# Remove the final row because it has no future value
df = df.dropna(subset=["future_congestion"])

# Convert target to integer
df["future_congestion"] = df["future_congestion"].astype(int)

# Select ML features
ml_columns = [
    "frame_number",
    "video_time_seconds",
    "vehicle_count",
    "average_speed_kmh",
    "congestion_score",
    "density_encoded",
    "congestion_target",
    "future_congestion"
]

ml_df = df[ml_columns].copy()

# Create output directory if necessary
os.makedirs("outputs", exist_ok=True)

# Save prepared dataset
ml_df.to_csv(OUTPUT_PATH, index=False)

print()
print("================================")
print("      PREPARED ML DATA")
print("================================")

print(f"Rows: {ml_df.shape[0]}")
print(f"Columns: {ml_df.shape[1]}")

print()
print("ML Features:")

for column in ml_columns:
    print(f"- {column}")

print()
print("================================")
print("     TARGET DISTRIBUTION")
print("================================")

print(
    ml_df["future_congestion"].value_counts().sort_index()
)

print()
print("Target Encoding:")
print("0 = NORMAL")
print("1 = MODERATE")
print("2 = HEAVY")

print()
print("================================")
print("       FIRST 10 ROWS")
print("================================")

print(
    ml_df.head(10).to_string(index=False)
)

print()
print("================================")
print("      ML DATA SAVED")
print("================================")

print(f"Saved to: {OUTPUT_PATH}")

print()
print("ML data preparation completed!")
print("================================")