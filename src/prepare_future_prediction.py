import pandas as pd

INPUT_PATH = "outputs/traffic_history_1sec.csv"
OUTPUT_PATH = "outputs/future_traffic_data.csv"

PREDICTION_HORIZON = 5


print("AI Traffic Intelligence - Future Traffic Prediction")
print()
print("Loading historical traffic data...")

df = pd.read_csv(INPUT_PATH)

print("Historical dataset loaded successfully!")

print()
print("================================")
print("      ORIGINAL DATASET")
print("================================")

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# Encode congestion level

congestion_mapping = {
    "NORMAL": 0,
    "MODERATE": 1,
    "HEAVY": 2
}

df["congestion_target"] = (
    df["congestion_level"]
    .map(congestion_mapping)
)


# Create future congestion target

df["future_congestion"] = (
    df["congestion_target"]
    .shift(-PREDICTION_HORIZON)
)


# Remove rows without a future value

df = df.dropna(
    subset=["future_congestion"]
)


df["future_congestion"] = (
    df["future_congestion"]
    .astype(int)
)


# Select features

FEATURES = [
    "vehicle_count",
    "average_speed_kmh",
    "congestion_score",
    "congestion_target"
]


OUTPUT_COLUMNS = [
    "video_time_seconds",
    "vehicle_count",
    "average_speed_kmh",
    "congestion_score",
    "congestion_target",
    "future_congestion"
]


ml_df = df[
    OUTPUT_COLUMNS
].copy()


# Save dataset

ml_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("================================")
print("    FUTURE PREDICTION DATA")
print("================================")

print(
    f"Prediction horizon: "
    f"{PREDICTION_HORIZON} seconds"
)

print(
    f"Rows available: "
    f"{len(ml_df)}"
)

print()
print("Features used:")

for feature in FEATURES:
    print(f"- {feature}")


print()
print("================================")
print("    FUTURE TARGET DISTRIBUTION")
print("================================")

print(
    ml_df[
        "future_congestion"
    ]
    .value_counts()
    .sort_index()
)


print()
print("Target encoding:")

print("0 = NORMAL")
print("1 = MODERATE")
print("2 = HEAVY")


print()
print("================================")
print("        FIRST 10 ROWS")
print("================================")

print(
    ml_df
    .head(10)
    .to_string(index=False)
)


print()
print("================================")
print("      DATASET SAVED")
print("================================")

print(
    f"Saved to: {OUTPUT_PATH}"
)

print()
print("Future traffic prediction dataset created!")
print("================================")