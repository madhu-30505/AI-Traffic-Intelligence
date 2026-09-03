import pandas as pd
import os


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_PATH = "outputs/traffic_history_detailed.csv"

OUTPUT_PATH = "outputs/traffic_ml_features.csv"

ROLLING_WINDOW = 30

FUTURE_SECONDS = 5

FPS = 30


# ==========================================
# START
# ==========================================

print("================================")
print(" AI Traffic Intelligence")
print(" Traffic Feature Engineering")
print("================================")
print()

print("Loading detailed traffic dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")

print()

print("Original rows:", len(df))


# ==========================================
# CONGESTION ENCODING
# ==========================================

congestion_mapping = {
    "NORMAL": 0,
    "MODERATE": 1,
    "HEAVY": 2
}

df["congestion_target"] = (
    df["congestion_level"]
    .map(congestion_mapping)
)


# ==========================================
# LAG FEATURES
# ==========================================

print()
print("Creating lag features...")


df["vehicle_count_lag_1"] = (
    df["vehicle_count"].shift(1)
)

df["vehicle_count_lag_3"] = (
    df["vehicle_count"].shift(3)
)

df["vehicle_count_lag_5"] = (
    df["vehicle_count"].shift(5)
)


df["speed_lag_1"] = (
    df["average_speed_kmh"].shift(1)
)

df["speed_lag_3"] = (
    df["average_speed_kmh"].shift(3)
)

df["speed_lag_5"] = (
    df["average_speed_kmh"].shift(5)
)


# ==========================================
# ROLLING FEATURES
# ==========================================

print("Creating rolling features...")


df["vehicle_count_rolling_mean"] = (
    df["vehicle_count"]
    .rolling(
        window=ROLLING_WINDOW
    )
    .mean()
)


df["speed_rolling_mean"] = (
    df["average_speed_kmh"]
    .rolling(
        window=ROLLING_WINDOW
    )
    .mean()
)


df["congestion_score_rolling_mean"] = (
    df["congestion_score"]
    .rolling(
        window=ROLLING_WINDOW
    )
    .mean()
)


# ==========================================
# TRAFFIC CHANGE FEATURES
# ==========================================

print("Creating traffic change features...")


df["vehicle_count_change"] = (
    df["vehicle_count"]
    - df["vehicle_count_lag_5"]
)


df["speed_change"] = (
    df["average_speed_kmh"]
    - df["speed_lag_5"]
)


# ==========================================
# FUTURE TARGET
# ==========================================

print("Creating future congestion target...")


FUTURE_FRAMES = (
    FUTURE_SECONDS * FPS
)


df["future_congestion"] = (
    df["congestion_target"]
    .shift(-FUTURE_FRAMES)
)


# ==========================================
# REMOVE INVALID ROWS
# ==========================================

df = df.dropna().reset_index(
    drop=True
)


# ==========================================
# SAVE DATASET
# ==========================================

os.makedirs(
    "outputs",
    exist_ok=True
)


df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# SUMMARY
# ==========================================

print()
print("================================")
print(" FEATURE ENGINEERING COMPLETED")
print("================================")

print()

print(
    f"Original rows: 1170"
)

print(
    f"Final rows: {len(df)}"
)

print()

print("Features created:")

features = [
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

for feature in features:

    print(
        f"- {feature}"
    )


print()

print("Future target:")

print(
    "- future_congestion"
)

print()

print("Future prediction horizon:")

print(
    f"- {FUTURE_SECONDS} seconds"
)

print()

print("Future congestion distribution:")

print(
    df[
        "future_congestion"
    ]
    .value_counts()
    .sort_index()
)

print()

print(
    f"CSV File: {OUTPUT_PATH}"
)

print()

print(
    "Traffic ML feature dataset created successfully!"
)

print("================================")
