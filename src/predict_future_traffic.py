import pandas as pd
import joblib
import os


DATA_PATH = "outputs/traffic_1sec_ml_features.csv"
MODEL_PATH = "models/traffic_forecasting_model.pkl"
OUTPUT_PATH = "outputs/future_traffic_predictions.csv"


print("======================================")
print(" AI Traffic Intelligence")
print(" Future Traffic Prediction")
print("======================================")
print()


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("Loading traffic ML data...")

df = pd.read_csv(DATA_PATH)

print(f"Rows available: {len(df)}")
print()


# --------------------------------------------------
# 2. Feature list
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


# --------------------------------------------------
# 3. Load trained model
# --------------------------------------------------

print("Loading trained Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print()


# --------------------------------------------------
# 4. Select latest traffic observation
# --------------------------------------------------

latest = df.iloc[-1]

latest_features = (
    df[features]
    .iloc[-1:]
)


current_time = latest["video_time_seconds"]


# --------------------------------------------------
# 5. Make prediction
# --------------------------------------------------

prediction = model.predict(
    latest_features
)[0]


# --------------------------------------------------
# 6. Prediction probabilities
# --------------------------------------------------

probabilities = model.predict_proba(
    latest_features
)[0]

classes = model.classes_


probability_dict = {}

for class_value, probability in zip(
    classes,
    probabilities
):

    probability_dict[int(class_value)] = (
        float(probability)
    )


# --------------------------------------------------
# 7. Convert prediction to label
# --------------------------------------------------

label_mapping = {
    0: "NORMAL",
    1: "MODERATE",
    2: "HEAVY"
}


predicted_level = (
    label_mapping[int(prediction)]
)


# --------------------------------------------------
# 8. Current traffic information
# --------------------------------------------------

current_vehicle_count = (
    latest["vehicle_count"]
)

current_speed = (
    latest["average_speed_kmh"]
)

current_score = (
    latest["congestion_score"]
)


# --------------------------------------------------
# 9. Prediction time
# --------------------------------------------------

prediction_time = (
    current_time + 5
)


# --------------------------------------------------
# 10. Save prediction
# --------------------------------------------------

prediction_row = pd.DataFrame([{

    "current_time_seconds":
        current_time,

    "prediction_time_seconds":
        prediction_time,

    "current_vehicle_count":
        current_vehicle_count,

    "current_average_speed_kmh":
        current_speed,

    "current_congestion_score":
        current_score,

    "predicted_congestion":
        predicted_level,

    "normal_probability":
        probability_dict.get(0, 0),

    "moderate_probability":
        probability_dict.get(1, 0),

    "heavy_probability":
        probability_dict.get(2, 0)

}])


os.makedirs(
    "outputs",
    exist_ok=True
)


prediction_row.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 11. Display prediction
# --------------------------------------------------

print("======================================")
print(" CURRENT TRAFFIC")
print("======================================")

print(
    f"Time: {current_time:.0f} seconds"
)

print(
    f"Vehicles: {current_vehicle_count:.2f}"
)

print(
    f"Average Speed: {current_speed:.2f} km/h"
)

print(
    f"Congestion Score: {current_score:.2f}"
)

print()


print("======================================")
print(" FUTURE TRAFFIC PREDICTION")
print("======================================")

print(
    f"Prediction time: "
    f"{prediction_time:.0f} seconds"
)

print(
    f"Predicted congestion: "
    f"{predicted_level}"
)

print()


print("Prediction probabilities:")

print(
    f"NORMAL   : "
    f"{probability_dict.get(0, 0) * 100:.2f}%"
)

print(
    f"MODERATE : "
    f"{probability_dict.get(1, 0) * 100:.2f}%"
)

print(
    f"HEAVY    : "
    f"{probability_dict.get(2, 0) * 100:.2f}%"
)

print()


print("======================================")
print(" PREDICTION SAVED")
print("======================================")

print(
    f"Output file: {OUTPUT_PATH}"
)

print()

print(
    "Future traffic prediction completed!"
)

print(
    "Note: prediction quality is limited by the"
)

print(
    "small training dataset."
)

print("======================================")
