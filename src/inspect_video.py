import cv2

VIDEO_PATH = "data/videos/traffic.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = total_frames / fps if fps > 0 else 0

print("================================")
print("       VIDEO INFORMATION")
print("================================")
print(f"Width:         {width}")
print(f"Height:        {height}")
print(f"FPS:           {fps:.2f}")
print(f"Total Frames:  {total_frames}")
print(f"Duration:      {duration:.2f} seconds")
print("================================")

cap.release()