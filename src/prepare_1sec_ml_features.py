import pandas as pd
import os

INPUT_PATH = "outputs/traffic_1sec_dataset.csv"
OUTPUT_PATH = "outputs/traffic_1sec_ml_features.csv"

FORECAST_HORIZON = 5


print("======================================")
print(" AI Traffic Intelligence")
print(" 1-Second ML Feature Engineering")
print("======================================")
print()


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("Loading 1-second traffic dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")
print(f"Rows: {len(df)}")
print()


# --------------------------------------------------
# 2. Sort by time
# --------------------------------------------------

df = df.sort_values(
    "video_time_seconds"
).reset_index(drop=True)


# --------------------------------------------------
# 3. Create lag features
# --------------------------------------------------

print("Creating lag features...")

# Vehicle count
df["vehicle_count_lag_1"] = (
    df["vehicle_count"].shift(1)
)

df["vehicle_count_lag_2"] = (
    df["vehicle_count"].shift(2)
)

df["vehicle_count_lag_3"] = (
    df["vehicle_count"].shift(3)
)

df["vehicle_count_lag_5"] = (
    df["vehicle_count"].shift(5)
)


# Speed
df["speed_lag_1"] = (
    df["average_speed_kmh"].shift(1)
)

df["speed_lag_2"] = (
    df["average_speed_kmh"].shift(2)
)

df["speed_lag_3"] = (
    df["average_speed_kmh"].shift(3)
)

df["speed_lag_5"] = (
    df["average_speed_kmh"].shift(5)
)


# Congestion score
df["congestion_score_lag_1"] = (
    df["congestion_score"].shift(1)
)

df["congestion_score_lag_3"] = (
    df["congestion_score"].shift(3)
)

df["congestion_score_lag_5"] = (
    df["congestion_score"].shift(5)
)


# --------------------------------------------------
# 4. Rolling features
# --------------------------------------------------

print("Creating rolling features...")

df["vehicle_count_rolling_3"] = (
    df["vehicle_count"]
    .rolling(window=3)
    .mean()
)

df["vehicle_count_rolling_5"] = (
    df["vehicle_count"]
    .rolling(window=5)
    .mean()
)


df["speed_rolling_3"] = (
    df["average_speed_kmh"]
    .rolling(window=3)
    .mean()
)

df["speed_rolling_5"] = (
    df["average_speed_kmh"]
    .rolling(window=5)
    .mean()
)


df["congestion_score_rolling_3"] = (
    df["congestion_score"]
    .rolling(window=3)
    .mean()
)

df["congestion_score_rolling_5"] = (
    df["congestion_score"]
    .rolling(window=5)
    .mean()
)


# --------------------------------------------------
# 5. Traffic change features
# --------------------------------------------------

print("Creating traffic trend features...")

df["vehicle_count_change"] = (
    df["vehicle_count"]
    - df["vehicle_count_lag_1"]
)

df["speed_change"] = (
    df["average_speed_kmh"]
    - df["speed_lag_1"]
)


# --------------------------------------------------
# 6. Create future target
# --------------------------------------------------

print(
    f"Creating {FORECAST_HORIZON}-second future target..."
)

df["future_congestion"] = (
    df["congestion_target"]
    .shift(-FORECAST_HORIZON)
)


# --------------------------------------------------
# 7. Remove rows with missing values
# --------------------------------------------------

df = df.dropna().reset_index(drop=True)


# --------------------------------------------------
# 8. Select ML features
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


output_columns = (
    ["video_time_seconds"]
    + features
    + ["future_congestion"]
)


ml_df = df[output_columns].copy()


# --------------------------------------------------
# 9. Convert target to integer
# --------------------------------------------------

ml_df["future_congestion"] = (
    ml_df["future_congestion"]
    .astype(int)
)


# --------------------------------------------------
# 10. Save dataset
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

ml_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 11. Display results
# --------------------------------------------------

print()
print("======================================")
print(" ML FEATURE DATASET CREATED")
print("======================================")
print()

print(f"Original rows: {39}")
print(f"Final ML rows: {len(ml_df)}")
print()

print("Number of features:")
print(len(features))
print()

print("Future congestion distribution:")

print(
    ml_df["future_congestion"]
    .value_counts()
    .sort_index()
)

print()

print("Feature columns:")

for feature in features:
    print(f"- {feature}")

print()

print("First 5 rows:")
print(
    ml_df.head().to_string(index=False)
)

print()
print(f"Saved to: {OUTPUT_PATH}")
print()

print(
    "1-second ML feature engineering completed successfully!"
)

print("======================================")
