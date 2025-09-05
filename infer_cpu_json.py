from ultralytics import YOLO
import cv2, json, datetime

model = YOLO("runs/detect/train3/weights/best.pt")

cap = cv2.VideoCapture(0)
frame_id = 0
all_detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    detections = []
    for r in results[0].boxes:
        box = r.xyxy[0].tolist()
        detections.append({
            "class": model.names[int(r.cls[0])],
            "confidence": float(r.conf[0]),
            "bounding_box": {
                "xmin": int(box[0]),
                "ymin": int(box[1]),
                "xmax": int(box[2]),
                "ymax": int(box[3])
            }
        })
        # Draw bounding box on frame
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
        cv2.putText(frame, f"{model.names[int(r.cls[0])]} {float(r.conf[0]):.2f}", 
                    (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    frame_data = {
        "frame_id": frame_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "detections": detections
    }
    all_detections.append(frame_data)

    cv2.imshow("Pothole Detection", frame)
    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

# Save JSON
with open("inference_results.json", "w") as f:
    json.dump(all_detections, f, indent=2)

print("Saved all detections to inference_results.json")
