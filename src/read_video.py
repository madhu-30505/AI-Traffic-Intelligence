import cv2


VIDEO_PATH = "data/test_traffic.mp4"


# Open the video
video = cv2.VideoCapture(VIDEO_PATH)


# Check whether the video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()


# Read video frame by frame
while True:

    success, frame = video.read()

    # Stop when there are no more frames
    if not success:
        break

    # Display the current frame
    cv2.imshow("Traffic Video", frame)

    # Press Q to stop
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


# Release the video
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Video processing completed.")