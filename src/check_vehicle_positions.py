from ultralytics import YOLO
import cv2

VIDEO_PATH = "data/videos/traffic.mp4"

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_number = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[1, 2, 3, 5, 7],
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

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

            x1, y1, x2, y2 = map(int, box)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            class_name = model.names[class_id]

            print(
                f"Frame: {frame_number} | "
                f"ID: {track_id} | "
                f"Class: {class_name} | "
                f"Center Y: {center_y}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            label = f"{class_name} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    cv2.imshow(
        "Vehicle Position Test",
        frame
    )

    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nPosition test completed.")
