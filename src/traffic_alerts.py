import pandas as pd
import os


INPUT_PATH = "outputs/future_traffic_predictions.csv"
OUTPUT_PATH = "outputs/traffic_alerts.csv"


print("======================================")
print(" AI Traffic Intelligence")
print(" Traffic Alert & Recommendation Engine")
print("======================================")
print()


# --------------------------------------------------
# 1. Load prediction
# --------------------------------------------------

print("Loading future traffic prediction...")

df = pd.read_csv(INPUT_PATH)

prediction = df.iloc[0]

print("Prediction loaded successfully!")
print()


# --------------------------------------------------
# 2. Read values
# --------------------------------------------------

current_time = prediction[
    "current_time_seconds"
]

prediction_time = prediction[
    "prediction_time_seconds"
]

current_vehicles = prediction[
    "current_vehicle_count"
]

current_speed = prediction[
    "current_average_speed_kmh"
]

current_score = prediction[
    "current_congestion_score"
]

predicted_level = prediction[
    "predicted_congestion"
]

normal_probability = prediction[
    "normal_probability"
]

moderate_probability = prediction[
    "moderate_probability"
]

heavy_probability = prediction[
    "heavy_probability"
]


# --------------------------------------------------
# 3. Generate alert
# --------------------------------------------------

if predicted_level == "NORMAL":

    alert_level = "LOW"

    alert_message = (
        "Traffic is expected to remain normal."
    )

    recommendation = (
        "No immediate traffic intervention required."
    )


elif predicted_level == "MODERATE":

    alert_level = "MEDIUM"

    alert_message = (
        "Moderate congestion is expected "
        "within the next 5 seconds."
    )

    recommendation = (
        "Monitor traffic closely and consider "
        "alternate routes if congestion increases."
    )


else:

    alert_level = "HIGH"

    alert_message = (
        "Heavy congestion is expected "
        "within the next 5 seconds."
    )

    recommendation = (
        "Immediate traffic management is recommended. "
        "Consider alternate routing and signal optimization."
    )


# --------------------------------------------------
# 4. Additional traffic insight
# --------------------------------------------------

if current_speed < 30:

    speed_insight = (
        "Vehicle speed is low, indicating slow traffic flow."
    )

elif current_speed < 50:

    speed_insight = (
        "Vehicle speed is moderate."
    )

else:

    speed_insight = (
        "Vehicle speed is relatively high."
    )


if current_vehicles >= 10:

    volume_insight = (
        "High vehicle volume detected."
    )

elif current_vehicles >= 5:

    volume_insight = (
        "Moderate vehicle volume detected."
    )

else:

    volume_insight = (
        "Low vehicle volume detected."
    )


# --------------------------------------------------
# 5. Create alert record
# --------------------------------------------------

alert_data = pd.DataFrame([{

    "current_time_seconds":
        current_time,

    "prediction_time_seconds":
        prediction_time,

    "current_vehicle_count":
        current_vehicles,

    "current_speed_kmh":
        current_speed,

    "current_congestion_score":
        current_score,

    "predicted_congestion":
        predicted_level,

    "model_probability":
        max(
            normal_probability,
            moderate_probability,
            heavy_probability
        ),

    "alert_level":
        alert_level,

    "alert_message":
        alert_message,

    "recommendation":
        recommendation,

    "speed_insight":
        speed_insight,

    "volume_insight":
        volume_insight

}])


# --------------------------------------------------
# 6. Save alerts
# --------------------------------------------------

os.makedirs(
    "outputs",
    exist_ok=True
)

alert_data.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 7. Display result
# --------------------------------------------------

print("======================================")
print(" TRAFFIC ALERT")
print("======================================")

print(
    f"Current time: "
    f"{current_time:.0f} seconds"
)

print(
    f"Prediction time: "
    f"{prediction_time:.0f} seconds"
)

print()

print(
    f"Predicted congestion: "
    f"{predicted_level}"
)

print(
    f"Alert level: "
    f"{alert_level}"
)

print()

print(
    "Alert:"
)

print(alert_message)

print()

print(
    "Recommendation:"
)

print(recommendation)

print()

print(
    "Traffic insights:"
)

print(
    f"- {speed_insight}"
)

print(
    f"- {volume_insight}"
)

print()

print("======================================")
print(" ALERT GENERATED SUCCESSFULLY")
print("======================================")

print(
    f"Saved to: {OUTPUT_PATH}"
)

print("======================================")
