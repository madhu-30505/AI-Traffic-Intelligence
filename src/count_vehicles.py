from ultralytics import YOLO
import cv2

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "data/videos/traffic.mp4"

MODEL_PATH = "yolo11n.pt"

# Vehicle classes from COCO dataset
# 2 = car
# 3 = motorcycle
# 1 = bicycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = [1, 2, 3, 5, 7]

# Counting line
LINE_Y = 650

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

# ============================================================
# TRACKING / COUNTING VARIABLES
# ============================================================

# IDs that have already been counted
counted_ids = set()

# Previous Y position of every tracked object
previous_positions = {}

# Vehicle counts
vehicle_counts = {
    "car": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "bus": 0,
    "truck": 0
}

total_count = 0

# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO DETECTION + TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=VEHICLE_CLASSES,
        conf=0.40,
        verbose=False
    )

    # --------------------------------------------------------
    # CHECK TRACKING IDs
    # --------------------------------------------------------

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

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

        # ----------------------------------------------------
        # PROCESS EACH VEHICLE
        # ----------------------------------------------------

        for box, track_id, class_id in zip(
            boxes,
            track_ids,
            class_ids
        ):

            x1, y1, x2, y2 = map(int, box)

            # Calculate center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            class_name = model.names[class_id]

            # ------------------------------------------------
            # COUNT VEHICLE CROSSING
            # ------------------------------------------------

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                # Vehicle moving DOWNWARD
                crossed_down = (
                    previous_y < LINE_Y
                    and center_y >= LINE_Y
                )

                # Vehicle moving UPWARD
                crossed_up = (
                    previous_y > LINE_Y
                    and center_y <= LINE_Y
                )

                if (
                    (crossed_down or crossed_up)
                    and track_id not in counted_ids
                ):

                    counted_ids.add(track_id)

                    total_count += 1

                    if class_name in vehicle_counts:
                        vehicle_counts[class_name] += 1

                    print(
                        f"Vehicle crossed | "
                        f"ID: {track_id} | "
                        f"Class: {class_name} | "
                        f"Total: {total_count}"
                    )

            # ------------------------------------------------
            # SAVE CURRENT POSITION
            # ------------------------------------------------

            previous_positions[track_id] = center_y

            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # DRAW CENTER POINT
            # ------------------------------------------------

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # ------------------------------------------------
            # DRAW VEHICLE LABEL
            # ------------------------------------------------

            label = (
                f"{class_name} "
                f"ID:{track_id}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # ========================================================
    # DRAW COUNTING LINE
    # ========================================================

    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (255, 0, 0),
        3
    )

    cv2.putText(
        frame,
        "COUNTING LINE",
        (30, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY TRAFFIC SUMMARY
    # ========================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (350, 220),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Total Vehicles: {total_count}",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {vehicle_counts['car']}",
        (35, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {vehicle_counts['motorcycle']}",
        (35, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Bicycles: {vehicle_counts['bicycle']}",
        (35, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {vehicle_counts['bus']}",
        (35, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {vehicle_counts['truck']}",
        (35, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(
        "AI Traffic Intelligence - Vehicle Counting",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("================================")
print("       TRAFFIC SUMMARY")
print("================================")

print(f"Cars:        {vehicle_counts['car']}")
print(f"Motorcycles: {vehicle_counts['motorcycle']}")
print(f"Bicycles:    {vehicle_counts['bicycle']}")
print(f"Buses:       {vehicle_counts['bus']}")
print(f"Trucks:      {vehicle_counts['truck']}")

print("--------------------------------")

print(f"Total Vehicles: {total_count}")

print("================================")