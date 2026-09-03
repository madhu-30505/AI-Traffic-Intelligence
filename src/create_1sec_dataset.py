import pandas as pd
import os


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_PATH = "outputs/traffic_history_detailed.csv"

OUTPUT_PATH = "outputs/traffic_1sec_dataset.csv"

FPS = 30


# ==========================================
# START
# ==========================================

print("================================")
print(" AI Traffic Intelligence")
print(" Improved 1-Second Aggregation")
print("================================")
print()

print("Loading frame-level traffic data...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully!")

print()

print(
    f"Frame-level rows: {len(df)}"
)


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
# CREATE SECOND
# ==========================================

df["second"] = (
    df["frame_number"] // FPS
)


# ==========================================
# AGGREGATE TRAFFIC
# ==========================================

print()

print(
    "Aggregating frames into 1-second intervals..."
)

aggregated = (
    df.groupby("second")
    .agg({

        # Average traffic volume
        "vehicle_count": "mean",

        # Average vehicle speed
        "average_speed_kmh": "mean",

        # Keep the highest congestion score
        "congestion_score": "max",

        # Keep the highest congestion severity
        "congestion_target": "max"

    })
    .reset_index()
)


# ==========================================
# ROUND VALUES
# ==========================================

aggregated["vehicle_count"] = (
    aggregated["vehicle_count"]
    .round(2)
)

aggregated["average_speed_kmh"] = (
    aggregated["average_speed_kmh"]
    .round(2)

)

aggregated["congestion_score"] = (
    aggregated["congestion_score"]
    .round(2)
)


# ==========================================
# CONGESTION LEVEL
# ==========================================

reverse_mapping = {
    0: "NORMAL",
    1: "MODERATE",
    2: "HEAVY"
}


aggregated["congestion_target"] = (
    aggregated["congestion_target"]
    .astype(int)
)


aggregated["congestion_level"] = (
    aggregated["congestion_target"]
    .map(reverse_mapping)
)


# ==========================================
# VIDEO TIME
# ==========================================

aggregated["video_time_seconds"] = (
    aggregated["second"]
)


# ==========================================
# REORDER COLUMNS
# ==========================================

aggregated = aggregated[
    [
        "video_time_seconds",
        "vehicle_count",
        "average_speed_kmh",
        "congestion_score",
        "congestion_level",
        "congestion_target"
    ]
]


# ==========================================
# SAVE DATASET
# ==========================================

os.makedirs(
    "outputs",
    exist_ok=True
)


aggregated.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# SUMMARY
# ==========================================

print()

print("================================")
print(" 1-SECOND DATASET CREATED")
print("================================")

print()

print(
    f"Original frame rows: {len(df)}"
)

print(
    f"1-second rows: {len(aggregated)}"
)

print()

print("Congestion distribution:")

print(
    aggregated[
        "congestion_level"
    ]
    .value_counts()
)


print()

print("First 10 rows:")

print(
    aggregated
    .head(10)
    .to_string(index=False)
)


print()

print(
    f"CSV File: {OUTPUT_PATH}"
)

print()

print(
    "Improved 1-second traffic dataset created successfully!"
)

print("================================")