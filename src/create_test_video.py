import cv2
import numpy as np


# -----------------------------
# Video configuration
# -----------------------------

WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION = 10

TOTAL_FRAMES = FPS * DURATION

OUTPUT_PATH = "data/test_traffic.mp4"


# -----------------------------
# Vehicle configuration
# -----------------------------

vehicles = [
    {"x": 350, "y": 650, "width": 70, "height": 120, "speed": 4},
    {"x": 600, "y": 550, "width": 80, "height": 130, "speed": 5},
    {"x": 850, "y": 700, "width": 75, "height": 120, "speed": 3},
    {"x": 1050, "y": 500, "width": 85, "height": 140, "speed": 6},
]


# -----------------------------
# Create video writer
# -----------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    FPS,
    (WIDTH, HEIGHT)
)


# -----------------------------
# Generate frames
# -----------------------------

for frame_number in range(TOTAL_FRAMES):

    # Create road background
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Road
    frame[:] = (60, 60, 60)

    # Road boundaries
    cv2.line(
        frame,
        (100, 0),
        (100, HEIGHT),
        (255, 255, 255),
        5
    )

    cv2.line(
        frame,
        (1180, 0),
        (1180, HEIGHT),
        (255, 255, 255),
        5
    )

    # Lane markings
    lane_positions = [370, 640, 910]

    for x in lane_positions:

        for y in range(0, HEIGHT, 80):

            cv2.line(
                frame,
                (x, y),
                (x, y + 40),
                (255, 255, 255),
                4
            )

    # -------------------------
    # Draw vehicles
    # -------------------------

    for vehicle in vehicles:

        x = vehicle["x"]
        y = vehicle["y"]

        width = vehicle["width"]
        height = vehicle["height"]

        # Vehicle body
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            -1
        )

        # Vehicle windshield
        cv2.rectangle(
            frame,
            (x + 10, y + 15),
            (x + width - 10, y + 45),
            (200, 200, 200),
            -1
        )

        # Vehicle label
        cv2.putText(
            frame,
            "CAR",
            (x + 10, y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Move vehicle upward
        vehicle["y"] -= vehicle["speed"]

        # Reset vehicle when it leaves the screen
        if vehicle["y"] + height < 0:
            vehicle["y"] = HEIGHT + np.random.randint(50, 300)

    # Frame counter
    cv2.putText(
        frame,
        f"Frame: {frame_number + 1}/{TOTAL_FRAMES}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Write frame
    video_writer.write(frame)


# -----------------------------
# Release video writer
# -----------------------------

video_writer.release()

print("Test traffic video created successfully!")
print(f"Saved to: {OUTPUT_PATH}")