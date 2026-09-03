import cv2

VIDEO_PATH = "data/videos/traffic.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

success, frame = cap.read()

if not success:
    print("ERROR: Could not read first frame.")
    cap.release()
    exit()

height, width = frame.shape[:2]

# Draw vertical grid lines
for x in range(0, width, 200):

    cv2.line(
        frame,
        (x, 0),
        (x, height),
        (0, 0, 0),
        2
    )

    # White rectangle behind X coordinate
    cv2.rectangle(
        frame,
        (x + 2, 2),
        (x + 65, 35),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        frame,
        f"X:{x}",
        (x + 5, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

# Draw horizontal grid lines
for y in range(0, height, 200):

    cv2.line(
        frame,
        (0, y),
        (width, y),
        (0, 0, 0),
        2
    )

    # White rectangle behind Y coordinate
    cv2.rectangle(
        frame,
        (2, y + 2),
        (85, y + 38),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        frame,
        f"Y:{y}",
        (8, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

# Resolution information
cv2.rectangle(
    frame,
    (20, 50),
    (350, 90),
    (255, 255, 255),
    -1
)

cv2.putText(
    frame,
    f"Resolution: {width} x {height}",
    (30, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2
)

cv2.imshow("ROI Coordinate Reference", frame)

print("Video frame displayed.")
print("Press Q to close the window.")

while True:

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()