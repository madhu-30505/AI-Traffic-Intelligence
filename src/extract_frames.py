import cv2
import os


VIDEO_PATH = "data/test_traffic.mp4"
OUTPUT_FOLDER = "outputs/frames"


# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Open the video
video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()


frame_number = 0


while True:

    success, frame = video.read()

    if not success:
        break

    # Save every 30th frame
    if frame_number % 30 == 0:

        filename = os.path.join(
            OUTPUT_FOLDER,
            f"frame_{frame_number:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

    frame_number += 1


video.release()

print(f"\nTotal frames processed: {frame_number}")
print("Frame extraction completed.")