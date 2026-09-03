import cv2
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "data/videos/traffic.mp4"
MODEL_PATH = "yolo11n.pt"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]
CONFIDENCE = 0.30


# Minimum movement in pixels before
# assigning a direction
MOVEMENT_THRESHOLD = 3


# ==========================================
# START PROGRAM
# ==========================================

print("AI Traffic Intelligence - Direction Detection")

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
# STORE PREVIOUS POSITIONS
# ==========================================

previous_positions = {}


# ==========================================
# DIRECTION COUNTERS
# ==========================================

direction_counts = {
    "UP": 0,
    "DOWN": 0,
    "LEFT": 0,
    "RIGHT": 0
}


# Store IDs whose direction has
# already been counted

counted_directions = set()


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
            # DEFAULT DIRECTION
            # ==================================

            direction = "STATIONARY"


            # ==================================
            # COMPARE WITH PREVIOUS POSITION
            # ==================================

            if track_id in previous_positions:

                previous_x, previous_y = (
                    previous_positions[track_id]
                )


                dx = center_x - previous_x
                dy = center_y - previous_y


                # ==================================
                # CHECK MOVEMENT
                # ==================================

                if (
                    abs(dx) < MOVEMENT_THRESHOLD
                    and
                    abs(dy) < MOVEMENT_THRESHOLD
                ):

                    direction = "STATIONARY"


                else:

                    # Determine dominant movement

                    if abs(dx) > abs(dy):

                        if dx > 0:
                            direction = "RIGHT"

                        else:
                            direction = "LEFT"

                    else:

                        if dy > 0:
                            direction = "DOWN"

                        else:
                            direction = "UP"


            # ==================================
            # SAVE CURRENT POSITION
            # ==================================

            previous_positions[track_id] = (
                center_x,
                center_y
            )


            # ==================================
            # COUNT DIRECTION ONCE PER ID
            # ==================================

            direction_key = (
                track_id,
                direction
            )

            if (
                direction != "STATIONARY"
                and
                direction_key not in counted_directions
            ):

                direction_counts[direction] += 1

                counted_directions.add(
                    direction_key
                )

                print(
                    f"Vehicle Direction | "
                    f"ID: {track_id} | "
                    f"Class: {class_name} | "
                    f"Direction: {direction}"
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


            # ==================================
            # DRAW CENTER
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

            label = (
                f"{class_name} "
                f"ID:{track_id}"
            )

            cv2.putText(
                frame,
                label,
                (
                    int(x1),
                    max(int(y1) - 30, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            # ==================================
            # DIRECTION LABEL
            # ==================================

            direction_label = (
                f"Direction: {direction}"
            )

            cv2.putText(
                frame,
                direction_label,
                (
                    int(x1),
                    max(int(y1) - 8, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


    # ======================================
    # INFORMATION PANEL
    # ======================================

    cv2.rectangle(
        frame,
        (20, 20),
        (330, 155),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "DIRECTION ANALYSIS",
        (35, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"UP: {direction_counts['UP']}",
        (35, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"DOWN: {direction_counts['DOWN']}",
        (130, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"LEFT: {direction_counts['LEFT']}",
        (35, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"RIGHT: {direction_counts['RIGHT']}",
        (130, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW VIDEO
    # ======================================

    cv2.imshow(
        "AI Traffic Intelligence - Direction Detection",
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
print("   DIRECTION ANALYSIS SUMMARY")
print("================================")

print(f"UP:       {direction_counts['UP']}")
print(f"DOWN:     {direction_counts['DOWN']}")
print(f"LEFT:     {direction_counts['LEFT']}")
print(f"RIGHT:    {direction_counts['RIGHT']}")

print()
print("Direction detection completed!")
print("================================")
