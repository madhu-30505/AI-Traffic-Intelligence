from ultralytics import YOLO
import cv2

VIDEO_PATH = "data/videos/traffic.mp4"

model = YOLO("yolo11n.pt")

results = model.track(
    source=VIDEO_PATH,
    show=True,
    tracker="botsort.yaml",
    classes=[2, 5, 7],
    persist=True
)

print("Tracking completed!")