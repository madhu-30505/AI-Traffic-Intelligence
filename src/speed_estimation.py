import cv2
from ultralytics import YOLO
import math
from collections import defaultdict, deque


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "data/videos/traffic.mp4"
MODEL_PATH = "yolo11n.pt"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]
CONFIDENCE = 0.30

# Number of positions used for speed calculation
HISTORY_SIZE = 10

# Calibration assumption
CALIBRATION_DISTANCE_PIXELS = 600.0
CALIBRATION_DISTANCE_METERS = 20.0

METERS_PER_PIXEL = (
    CALIBRATION_DISTANCE_METERS
    / CALIBRATION_DISTANCE_PIXELS
)

# Sanity limits for this project
MIN_SPEED_KMH = 0.0
MAX_SPEED_KMH = 120.0


# ==========================================
# START
# ==========================================

print("AI Traffic Intelligence - Speed Estimation")
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

print("Traffic video opened successfully!")


# ==========================================
# VIDEO FPS
# ==========================================

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    print("ERROR: Invalid video FPS.")
    cap.release()
    exit()

print(f"Video FPS: {fps:.2f}")


# ==========================================
# POSITION HISTORY
# ==========================================

position_history = defaultdict(
    lambda: deque(maxlen=HISTORY_SIZE)
)


# ==========================================
# SPEED HISTORY
# ==========================================

speed_history = defaultdict(
    lambda: deque(maxlen=5)
)


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
    # PROCESS VEHICLES
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
            # CENTER POINT
            # ==================================

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            class_name = model.names[class_id]


            # ==================================
            # STORE POSITION
            # ==================================

            position_history[track_id].append(
                (center_x, center_y)
            )

            history = position_history[track_id]


            # ==================================
            # CALCULATE SPEED
            # ==================================

            estimated_speed = 0.0


            if len(history) >= 2:

                first_x, first_y = history[0]

                last_x, last_y = history[-1]


                # Distance in pixels

                pixel_distance = math.sqrt(
                    (last_x - first_x) ** 2
                    +
                    (last_y - first_y) ** 2
                )


                # Time elapsed

                frame_difference = len(history) - 1

                time_elapsed = (
                    frame_difference / fps
                )


                if time_elapsed > 0:

                    # Pixel speed

                    pixel_speed = (
                        pixel_distance
                        / time_elapsed
                    )


                    # Convert to meters/second

                    meter_speed = (
                        pixel_speed
                        * METERS_PER_PIXEL
                    )


                    # Convert to km/h

                    raw_speed = (
                        meter_speed
                        * 3.6
                    )


                    # ==================================
                    # SPEED SANITY FILTER
                    # ==================================

                    if (
                        raw_speed >= MIN_SPEED_KMH
                        and
                        raw_speed <= MAX_SPEED_KMH
                    ):

                        speed_history[track_id].append(
                            raw_speed
                        )


            # ==================================
            # SMOOTH SPEED
            # ==================================

            if len(speed_history[track_id]) > 0:

                estimated_speed = (
                    sum(speed_history[track_id])
                    /
                    len(speed_history[track_id])
                )


            # ==================================
            # DRAW BOUNDING BOX
            # ==================================

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )


            # ==================================
            # CENTER POINT
            # ==================================

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


            # ==================================
            # VEHICLE LABEL
            # ==================================

            vehicle_label = (
                f"{class_name} ID:{track_id}"
            )

            cv2.putText(
                frame,
                vehicle_label,
                (
                    int(x1),
                    max(int(y1) - 32, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            # ==================================
            # SPEED LABEL
            # ==================================

            speed_label = (
                f"Speed: {estimated_speed:.1f} km/h"
            )

            cv2.putText(
                frame,
                speed_label,
                (
                    int(x1),
                    max(int(y1) - 8, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            # ==================================
            # TERMINAL OUTPUT
            # ==================================

            if len(history) >= 2:

                print(
                    f"ID: {track_id} | "
                    f"{class_name} | "
                    f"Estimated Speed: "
                    f"{estimated_speed:.2f} km/h"
                )


    # ======================================
    # INFORMATION PANEL
    # ======================================

    cv2.rectangle(
        frame,
        (20, 20),
        (480, 125),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AI TRAFFIC SPEED ESTIMATION",
        (35, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Calibration: {CALIBRATION_DISTANCE_PIXELS:.0f}px"
        f" = {CALIBRATION_DISTANCE_METERS:.0f}m",
        (35, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Maximum Display Speed: {MAX_SPEED_KMH:.0f} km/h",
        (35, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW VIDEO
    # ======================================

    cv2.imshow(
        "AI Traffic Intelligence - Speed Estimation",
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
# SUMMARY
# ==========================================

print()
print("================================")
print("   SPEED ANALYSIS SUMMARY")
print("================================")
print(f"Video FPS: {fps:.2f}")
print(f"History Window: {HISTORY_SIZE} frames")
print(
    f"Calibration: "
    f"{CALIBRATION_DISTANCE_PIXELS:.0f} px "
    f"= {CALIBRATION_DISTANCE_METERS:.1f} m"
)
print(f"Maximum Speed Filter: {MAX_SPEED_KMH:.0f} km/h")
print("Speed Type: Filtered approximate km/h")
print()
print("Speed estimation completed!")
print("================================")