from ultralytics import YOLO
import cv2


IMAGE_PATH = "data/images/traffic.jpg"


# Load YOLO model
model = YOLO("yolo11n.pt")


# Run object detection
results = model(IMAGE_PATH)


# Process the results
for result in results:

    boxes = result.boxes

    print("\nDetected objects:")

    for box in boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        coordinates = box.xyxy[0].tolist()

        print(
            f"Object: {class_name} | "
            f"Confidence: {confidence:.2f} | "
            f"Box: {coordinates}"
        )


# Generate annotated image
annotated_image = results[0].plot()

cv2.imwrite(
    "outputs/yolo_detection.jpg",
    annotated_image
)

print("\nDetection completed!")
print("Saved: outputs/yolo_detection.jpg")