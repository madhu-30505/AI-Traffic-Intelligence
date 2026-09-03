import cv2
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "data/videos/traffic.mp4"
MODEL_PATH = "yolo11n.pt"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]
CONFIDENCE = 0.30


# ==========================================
# ZONE DEFINITIONS
# ==========================================

ZONE_1_Y1 = 200
ZONE_1_Y2 = 400

ZONE_2_Y1 = 400
ZONE_2_Y2 = 600

ZONE_3_Y1 = 600
ZONE_3_Y2 = 800


# ==========================================
# START
# ==========================================

print("AI Traffic Intelligence - Zone Analysis")

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
    # INITIALIZE ZONE COUNTS
    # ======================================

    zone_1_count = 0
    zone_2_count = 0
    zone_3_count = 0


    # ======================================
    # CHECK DETECTED VEHICLES
    # ======================================

    if results[0].boxes is not None:

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )


        for box in boxes:

            x1, y1, x2, y2 = box


            # ==================================
            # FIND VEHICLE CENTER
            # ==================================

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # ==================================
            # ASSIGN VEHICLE TO ZONE
            # ==================================

            if ZONE_1_Y1 <= center_y < ZONE_1_Y2:

                zone_1_count += 1

            elif ZONE_2_Y1 <= center_y < ZONE_2_Y2:

                zone_2_count += 1

            elif ZONE_3_Y1 <= center_y <= ZONE_3_Y2:

                zone_3_count += 1


    # ======================================
    # DETERMINE ZONE STATUS
    # ======================================

    if zone_1_count <= 5:
        zone_1_status = "LOW"

    elif zone_1_count <= 10:
        zone_1_status = "MEDIUM"

    else:
        zone_1_status = "HIGH"


    if zone_2_count <= 5:
        zone_2_status = "LOW"

    elif zone_2_count <= 10:
        zone_2_status = "MEDIUM"

    else:
        zone_2_status = "HIGH"


    if zone_3_count <= 5:
        zone_3_status = "LOW"

    elif zone_3_count <= 10:
        zone_3_status = "MEDIUM"

    else:
        zone_3_status = "HIGH"


    # ======================================
    # DRAW ZONES
    # ======================================

    cv2.rectangle(
        frame,
        (0, ZONE_1_Y1),
        (frame.shape[1], ZONE_1_Y2),
        (255, 0, 0),
        3
    )

    cv2.rectangle(
        frame,
        (0, ZONE_2_Y1),
        (frame.shape[1], ZONE_2_Y2),
        (0, 255, 0),
        3
    )

    cv2.rectangle(
        frame,
        (0, ZONE_3_Y1),
        (frame.shape[1], ZONE_3_Y2),
        (0, 0, 255),
        3
    )


    # ======================================
    # DISPLAY ZONE INFORMATION
    # ======================================

    cv2.putText(
        frame,
        f"ZONE 1: {zone_1_count} vehicles - {zone_1_status}",
        (30, ZONE_1_Y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"ZONE 2: {zone_2_count} vehicles - {zone_2_status}",
        (30, ZONE_2_Y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"ZONE 3: {zone_3_count} vehicles - {zone_3_status}",
        (30, ZONE_3_Y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY VIDEO
    # ======================================

    cv2.imshow(
        "AI Traffic Intelligence - Zone Analysis",
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
print("   ZONE ANALYSIS COMPLETED")
print("================================")

print("Zone 1: Road section 200 - 400")
print("Zone 2: Road section 400 - 600")
print("Zone 3: Road section 600 - 800")

print()
print("Zone-based traffic analysis completed!")

print("================================")
