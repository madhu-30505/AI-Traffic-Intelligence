import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import os


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "data/videos/traffic.mp4"
MODEL_PATH = "yolo11n.pt"
OUTPUT_PATH = "outputs/traffic_history_detailed.csv"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]

CONFIDENCE = 0.30

ROI_X1 = 200
ROI_Y1 = 200
ROI_X2 = 1400
ROI_Y2 = 800

HISTORY_SIZE = 10
SPEED_HISTORY_SIZE = 5

CALIBRATION_DISTANCE_PIXELS = 600.0
CALIBRATION_DISTANCE_METERS = 20.0

MIN_SPEED_KMH = 0.0
MAX_SPEED_KMH = 120.0


# ==========================================
# SPEED CALIBRATION
# ==========================================

METERS_PER_PIXEL = (
    CALIBRATION_DISTANCE_METERS
    / CALIBRATION_DISTANCE_PIXELS
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_speed(history, fps):

    if len(history) < 2:
        return None

    first_x, first_y = history[0]
    last_x, last_y = history[-1]

    pixel_distance = math.sqrt(
        (last_x - first_x) ** 2
        + (last_y - first_y) ** 2
    )

    frame_difference = len(history) - 1

    if frame_difference <= 0:
        return None

    time_elapsed = frame_difference / fps

    pixel_speed = pixel_distance / time_elapsed

    meter_speed = (
        pixel_speed
        * METERS_PER_PIXEL
    )

    speed_kmh = meter_speed * 3.6

    if (
        speed_kmh < MIN_SPEED_KMH
        or speed_kmh > MAX_SPEED_KMH
    ):
        return None

    return speed_kmh


def get_density(vehicle_count):

    if vehicle_count <= 5:
        return "LOW"

    elif vehicle_count <= 10:
        return "MEDIUM"

    else:
        return "HIGH"


def calculate_congestion(
    vehicle_count,
    average_speed
):

    count_score = 0

    if vehicle_count <= 5:
        count_score = 0

    elif vehicle_count <= 10:
        count_score = 1

    else:
        count_score = 2


    speed_score = 0

    if average_speed == 0:
        speed_score = 0

    elif average_speed < 30:
        speed_score = 2

    elif average_speed < 50:
        speed_score = 1

    else:
        speed_score = 0


    congestion_score = (
        count_score
        + speed_score
    )


    if congestion_score <= 1:
        congestion_level = "NORMAL"

    elif congestion_score <= 2:
        congestion_level = "MODERATE"

    else:
        congestion_level = "HEAVY"


    return congestion_score, congestion_level


# ==========================================
# START
# ==========================================

print("================================")
print(" AI Traffic Intelligence")
print(" Detailed Traffic Data Logger")
print("================================")
print()

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")
print()


# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("Error: Could not open traffic video.")

    exit()


fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print(
    f"Video FPS: {fps:.2f}"
)

print(
    f"Total Frames: {total_frames}"
)

print()


# ==========================================
# TRACKING HISTORY
# ==========================================

position_history = defaultdict(
    lambda: deque(
        maxlen=HISTORY_SIZE
    )
)

speed_history = defaultdict(
    lambda: deque(
        maxlen=SPEED_HISTORY_SIZE
    )
)


# ==========================================
# DATA STORAGE
# ==========================================

records = []


frame_number = 0


# ==========================================
# PROCESS VIDEO
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break


    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=VEHICLE_CLASSES,
        conf=CONFIDENCE,
        verbose=False
    )


    current_vehicle_count = 0

    current_speeds = []


    if (
        results
        and results[0].boxes
        and results[0].boxes.id is not None
    ):

        boxes = results[0].boxes

        track_ids = (
            boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        classes = (
            boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        coordinates = (
            boxes.xyxy
            .cpu()
            .numpy()
        )


        for track_id, class_id, box in zip(
            track_ids,
            classes,
            coordinates
        ):

            x1, y1, x2, y2 = box


            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ==================================
            # ROI CHECK
            # ==================================

            inside_roi = (
                ROI_X1 <= center_x <= ROI_X2
                and
                ROI_Y1 <= center_y <= ROI_Y2
            )


            if not inside_roi:
                continue


            current_vehicle_count += 1


            # ==================================
            # POSITION HISTORY
            # ==================================

            position_history[
                track_id
            ].append(
                (
                    center_x,
                    center_y
                )
            )


            # ==================================
            # SPEED
            # ==================================

            speed = calculate_speed(
                position_history[track_id],
                fps
            )


            if speed is not None:

                speed_history[
                    track_id
                ].append(speed)


            if len(
                speed_history[track_id]
            ) > 0:

                smoothed_speed = sum(
                    speed_history[track_id]
                ) / len(
                    speed_history[track_id]
                )

                current_speeds.append(
                    smoothed_speed
                )


    # ==========================================
    # AVERAGE SPEED
    # ==========================================

    if len(current_speeds) > 0:

        average_speed = (
            sum(current_speeds)
            / len(current_speeds)
        )

    else:

        average_speed = 0.0


    average_speed = round(
        average_speed,
        2
    )


    # ==========================================
    # DENSITY
    # ==========================================

    density = get_density(
        current_vehicle_count
    )


    # ==========================================
    # CONGESTION
    # ==========================================

    congestion_score, congestion_level = (
        calculate_congestion(
            current_vehicle_count,
            average_speed
        )
    )


    # ==========================================
    # VIDEO TIME
    # ==========================================

    video_time_seconds = (
        frame_number / fps
    )


    # ==========================================
    # STORE RECORD
    # ==========================================

    records.append({

        "video_time_seconds":
            round(
                video_time_seconds,
                2
            ),

        "frame_number":
            frame_number,

        "vehicle_count":
            current_vehicle_count,

        "average_speed_kmh":
            average_speed,

        "density":
            density,

        "congestion_score":
            congestion_score,

        "congestion_level":
            congestion_level
    })


    frame_number += 1


    # ==========================================
    # PROGRESS
    # ==========================================

    if frame_number % 100 == 0:

        print(
            f"Processed frames: {frame_number}"
        )


# ==========================================
# RELEASE VIDEO
# ==========================================

cap.release()


# ==========================================
# SAVE DATA
# ==========================================

df = pd.DataFrame(records)


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
print(" DATA COLLECTION COMPLETED")
print("================================")

print(
    f"Frames Processed: {frame_number}"
)

print(
    f"Rows Created: {len(df)}"
)

print()

print("Columns:")

for column in df.columns:

    print(
        f"- {column}"
    )

print()

print(
    "Congestion Distribution:"
)

print(
    df[
        "congestion_level"
    ].value_counts()
)

print()

print(
    f"CSV File: {OUTPUT_PATH}"
)

print()

print(
    "Detailed traffic dataset created successfully!"
)

print("================================")