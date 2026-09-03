import cv2
import math
from collections import defaultdict, deque

from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "data/videos/traffic.mp4"
MODEL_PATH = "yolo11n.pt"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]
CONFIDENCE = 0.30

HISTORY_SIZE = 10
SPEED_HISTORY_SIZE = 5

# Approximate calibration
# 600 pixels = 20 meters
CALIBRATION_DISTANCE_PIXELS = 600.0
CALIBRATION_DISTANCE_METERS = 20.0

METERS_PER_PIXEL = (
    CALIBRATION_DISTANCE_METERS
    / CALIBRATION_DISTANCE_PIXELS
)

# Speed sanity filter
MIN_SPEED_KMH = 0.0
MAX_SPEED_KMH = 120.0


# ==========================================
# ROI
# ==========================================

ROI_X1 = 200
ROI_Y1 = 200
ROI_X2 = 1400
ROI_Y2 = 800


# ==========================================
# CONGESTION THRESHOLDS
# ==========================================

LOW_VEHICLES = 5
MEDIUM_VEHICLES = 10

LOW_SPEED = 30.0
MEDIUM_SPEED = 50.0

CONGESTION_DURATION_THRESHOLD = 30


# ==========================================
# LOAD MODEL
# ==========================================

print("AI Traffic Intelligence - Improved Congestion")

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")


# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

print("Traffic video opened successfully!")
print(f"Video FPS: {fps:.2f}")


# ==========================================
# TRACKING HISTORY
# ==========================================

position_history = defaultdict(
    lambda: deque(maxlen=HISTORY_SIZE)
)

speed_history = defaultdict(
    lambda: deque(maxlen=SPEED_HISTORY_SIZE)
)


# ==========================================
# CONGESTION STATE
# ==========================================

congestion_frame_count = 0


# ==========================================
# PROCESS VIDEO
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # ======================================
    # YOLO TRACKING
    # ======================================

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=VEHICLE_CLASSES,
        conf=CONFIDENCE,
        verbose=False
    )


    # ======================================
    # VARIABLES
    # ======================================

    vehicle_count = 0
    valid_speeds = []


    # ======================================
    # PROCESS TRACKED VEHICLES
    # ======================================

    if results[0].boxes.id is not None:

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            results[0]
            .boxes
            .id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = (
            results[0]
            .boxes
            .cls
            .int()
            .cpu()
            .tolist()
        )


        for box, track_id, class_id in zip(
            boxes,
            track_ids,
            class_ids
        ):

            x1, y1, x2, y2 = box


            # ==================================
            # VEHICLE CENTER
            # ==================================

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # ==================================
            # CHECK ROI
            # ==================================

            inside_roi = (
                ROI_X1 <= center_x <= ROI_X2
                and
                ROI_Y1 <= center_y <= ROI_Y2
            )


            if inside_roi:

                vehicle_count += 1


            # ==================================
            # STORE POSITION HISTORY
            # ==================================

            history = position_history[track_id]

            history.append(
                (center_x, center_y)
            )


            # ==================================
            # SPEED CALCULATION
            # ==================================

            if len(history) >= 2:

                first_x, first_y = history[0]
                last_x, last_y = history[-1]


                pixel_distance = math.sqrt(
                    (last_x - first_x) ** 2
                    +
                    (last_y - first_y) ** 2
                )


                frame_difference = len(history) - 1


                time_elapsed = (
                    frame_difference / fps
                )


                if time_elapsed > 0:

                    pixel_speed = (
                        pixel_distance
                        /
                        time_elapsed
                    )


                    meter_speed = (
                        pixel_speed
                        *
                        METERS_PER_PIXEL
                    )


                    speed_kmh = (
                        meter_speed * 3.6
                    )


                    # ==================================
                    # FILTER UNREALISTIC VALUES
                    # ==================================

                    if (
                        MIN_SPEED_KMH
                        <= speed_kmh
                        <= MAX_SPEED_KMH
                    ):

                        speed_history[track_id].append(
                            speed_kmh
                        )


            # ==================================
            # GET SMOOTHED SPEED
            # ==================================

            if speed_history[track_id]:

                average_vehicle_speed = (
                    sum(speed_history[track_id])
                    /
                    len(speed_history[track_id])
                )

                if inside_roi:

                    valid_speeds.append(
                        average_vehicle_speed
                    )


            # ==================================
            # DRAW VEHICLE
            # ==================================

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )


            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1
            )


            class_name = model.names[class_id]


            # ==================================
            # DISPLAY VEHICLE INFORMATION
            # ==================================

            label = (
                f"{class_name} "
                f"ID:{track_id}"
            )

            cv2.putText(
                frame,
                label,
                (
                    int(x1),
                    max(int(y1) - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


    # ======================================
    # AVERAGE SPEED
    # ======================================

    if valid_speeds:

        average_speed = (
            sum(valid_speeds)
            /
            len(valid_speeds)
        )

    else:

        average_speed = 0.0


    # ======================================
    # DETERMINE DENSITY
    # ======================================

    if vehicle_count <= LOW_VEHICLES:

        density = "LOW"

    elif vehicle_count <= MEDIUM_VEHICLES:

        density = "MEDIUM"

    else:

        density = "HIGH"


    # ======================================
    # CONGESTION SCORE
    # ======================================

    congestion_score = 0


    # Vehicle count contribution

    if vehicle_count <= LOW_VEHICLES:

        congestion_score += 0

    elif vehicle_count <= MEDIUM_VEHICLES:

        congestion_score += 1

    else:

        congestion_score += 2


    # Speed contribution

    if average_speed == 0:

        speed_score = 0

    elif average_speed < LOW_SPEED:

        speed_score = 2

    elif average_speed < MEDIUM_SPEED:

        speed_score = 1

    else:

        speed_score = 0


    congestion_score += speed_score


    # ======================================
    # DETERMINE CONGESTION
    # ======================================

    if congestion_score <= 1:

        congestion = "NORMAL"

    elif congestion_score <= 2:

        congestion = "MODERATE"

    else:

        congestion = "HEAVY"


    # ======================================
    # DURATION TRACKING
    # ======================================

    if congestion in ["MODERATE", "HEAVY"]:

        congestion_frame_count += 1

    else:

        congestion_frame_count = 0


    congestion_duration = (
        congestion_frame_count / fps
    )


    # ======================================
    # PERSISTENT CONGESTION
    # ======================================

    if (
        congestion_duration
        >= CONGESTION_DURATION_THRESHOLD
    ):

        persistent_status = (
            "PERSISTENT CONGESTION"
        )

    else:

        persistent_status = (
            "TEMPORARY CONGESTION"
        )


    # ======================================
    # DRAW ROI
    # ======================================

    cv2.rectangle(
        frame,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (255, 0, 0),
        3
    )


    cv2.putText(
        frame,
        "TRAFFIC ROI",
        (
            ROI_X1 + 10,
            ROI_Y1 + 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ======================================
    # INFORMATION PANEL
    # ======================================

    cv2.rectangle(
        frame,
        (20, 20),
        (450, 235),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Density: {density}",
        (35, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Avg Speed: {average_speed:.1f} km/h",
        (35, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Score: {congestion_score}",
        (35, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Congestion: {congestion}",
        (35, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Duration: {congestion_duration:.1f}s",
        (230, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ======================================
    # PERSISTENT STATUS
    # ======================================

    cv2.putText(
        frame,
        persistent_status,
        (35, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "AI Traffic Intelligence - Improved Congestion",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()


# ==========================================
# FINAL SUMMARY
# ==========================================

print()
print("================================")
print("   IMPROVED CONGESTION SUMMARY")
print("================================")

print(
    f"Video FPS: {fps:.2f}"
)

print(
    f"ROI: "
    f"({ROI_X1}, {ROI_Y1}) -> "
    f"({ROI_X2}, {ROI_Y2})"
)

print(
    f"Speed Calibration: "
    f"{CALIBRATION_DISTANCE_PIXELS:.0f} px = "
    f"{CALIBRATION_DISTANCE_METERS:.1f} m"
)

print(
    f"Maximum Speed Filter: "
    f"{MAX_SPEED_KMH:.0f} km/h"
)

print()
print(
    "Congestion combines:"
)

print(
    "1. Vehicle count"
)

print(
    "2. Average speed"
)

print(
    "3. Congestion duration"
)

print()
print(
    "Improved congestion analysis completed!"
)

print("================================")