from ultralytics import YOLO
import cv2
from collections import deque


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "data/videos/traffic.mp4"
MODEL_PATH = "yolo11n.pt"

# Vehicle classes from COCO dataset
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

# YOLO confidence threshold
CONFIDENCE = 0.30

# Number of recent frames used for
# calculating the moving average
WINDOW_SIZE = 10

# ==========================================
# TRAFFIC ROI
# ==========================================

ROI_X1 = 200
ROI_Y1 = 200

ROI_X2 = 1400
ROI_Y2 = 800


# ==========================================
# LOAD YOLO MODEL
# ==========================================

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


# ==========================================
# MOVING AVERAGE STORAGE
# ==========================================

recent_counts = deque(maxlen=WINDOW_SIZE)


# ==========================================
# PROCESS VIDEO
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # ======================================
    # YOLO DETECTION
    # ======================================

    results = model(
        frame,
        classes=VEHICLE_CLASSES,
        conf=CONFIDENCE,
        verbose=False
    )


    # ======================================
    # COUNT VEHICLES INSIDE ROI
    # ======================================

    vehicle_count = 0

    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = box

            # Calculate center of bounding box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2


            # Check whether vehicle center
            # is inside the traffic ROI

            if (
                ROI_X1 <= center_x <= ROI_X2
                and
                ROI_Y1 <= center_y <= ROI_Y2
            ):

                vehicle_count += 1


    # ======================================
    # UPDATE MOVING AVERAGE
    # ======================================

    recent_counts.append(vehicle_count)

    average_count = (
        sum(recent_counts) / len(recent_counts)
    )


    # ======================================
    # TRAFFIC DENSITY
    # ======================================

    if average_count <= 10:

        density = "LOW"

    elif average_count <= 20:

        density = "MEDIUM"

    else:

        density = "HIGH"


    # ======================================
    # DISPLAY TERMINAL INFORMATION
    # ======================================

    print(
        f"Current: {vehicle_count} | "
        f"Average: {average_count:.1f} | "
        f"Traffic Density: {density}"
    )


    # ======================================
    # DRAW YOLO DETECTIONS
    # ======================================

    annotated_frame = results[0].plot()


    # ======================================
    # DRAW TRAFFIC ROI
    # ======================================

    cv2.rectangle(
        annotated_frame,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (255, 0, 0),
        3
    )


    cv2.putText(
        annotated_frame,
        "TRAFFIC ROI",
        (ROI_X1 + 10, ROI_Y1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY CURRENT VEHICLE COUNT
    # ======================================

    cv2.putText(
        annotated_frame,
        f"Current Vehicles: {vehicle_count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY AVERAGE VEHICLE COUNT
    # ======================================

    cv2.putText(
        annotated_frame,
        f"Average Vehicles: {average_count:.1f}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY TRAFFIC DENSITY
    # ======================================

    cv2.putText(
        annotated_frame,
        f"Traffic Density: {density}",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY VIDEO
    # ======================================

    cv2.imshow(
        "AI Traffic Intelligence - Traffic Density",
        annotated_frame
    )


    # ======================================
    # PRESS Q TO EXIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# RELEASE RESOURCES
# ==========================================

cap.release()

cv2.destroyAllWindows()


# ==========================================
# FINAL SUMMARY
# ==========================================

print()

print("================================")
print("   TRAFFIC DENSITY SUMMARY")
print("================================")

print(f"ROI:")
print(f"  X: {ROI_X1} → {ROI_X2}")
print(f"  Y: {ROI_Y1} → {ROI_Y2}")

print()

print(f"Window Size: {WINDOW_SIZE} frames")
print(f"Confidence: {CONFIDENCE}")

print()

print("Traffic density analysis completed!")

print("================================")