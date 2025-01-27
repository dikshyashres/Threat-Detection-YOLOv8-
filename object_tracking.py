import cv2
import time
from ultralytics import YOLO

# Define class names for both models
CLASS_NAMES_1 = ['gun', 'knife']  # Classes for best.pt
CLASS_NAMES_2 = ['fight','no fight']  # Classes for fight.pt

def show_fps(frame, fps):
    """Display FPS on the frame."""
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def main():
    # Load both YOLO models
    model1 = YOLO(r'C:\threat-detection-system\weapons.pt')  # Path to best.pt
    model2 = YOLO(r'C:\threat-detection-system\fight.pt')  # Path to fight.pt

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Perform inference with the first model (best.pt)
        results1 = model1(frame)

        # Perform inference with the second model (fight.pt)
        results2 = model2(frame)

        # Annotate frame with detections from the first model
        for result in results1:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                confidence = float(box.conf[0])  # Confidence score
                class_id = int(box.cls[0])  # Class ID

                if confidence >= 0.3:  # Confidence threshold
                    label = f"{CLASS_NAMES_1[class_id]}: {confidence:.2f}"
                    color = (0, 255, 0)  # Green color for bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Annotate frame with detections from the second model
        for result in results2:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                confidence = float(box.conf[0])  # Confidence score
                class_id = int(box.cls[0])  # Class ID

                if confidence >= 0.3:  # Confidence threshold
                    label = f"{CLASS_NAMES_2[class_id]}: {confidence:.2f}"
                    color = (0, 0, 255)  # Red color for bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        show_fps(frame, fps)

        # Show the frame
        cv2.imshow("YOLO Multi-Model Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
