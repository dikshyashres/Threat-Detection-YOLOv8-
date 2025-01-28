import cv2
from ultralytics import YOLO  # Import YOLO from the ultralytics package

# Path to your trained YOLOv11 model
model_path = 'C:\threat-detection-system\weapons.pt'  # Path to your custom YOLOv11 model

# Load the trained YOLO model
model = YOLO(model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera; use 1 or 2 for other cameras

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform inference on the frame
    results = model.predict(frame)  # Correct method to run inference

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()  # Plot results onto the frame

    # Display the annotated frame
    cv2.imshow("YOLOv11 Real-Time Detection", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

