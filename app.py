import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load YOLO models
weapon_model = YOLO(r'C:\threat-detection-system\weapons.pt')  # Path to weapons.pt
fight_model = YOLO(r'C:\threat-detection-system\fight.pt')  # Path to fight.pt

# Define global variables
cap = None
alert_message = "No Threat Detected"  # Default alert message

# Helper function to draw bounding boxes
def draw_boxes(frame, results, model):
    for result in results:
        for box in result.boxes:
            if box.conf[0] >= 0.7:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = float(box.conf[0])  # Confidence score
                class_id = int(box.cls[0])  # Class ID
                class_name = model.names.get(class_id, "Unknown")  # Get class name

                # Draw bounding box
                color = (0, 255, 0) if "gun" in class_name or "knife" in class_name else (0, 0, 255)
                label = f"{class_name}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def generate_frames():
    """Capture frames, run YOLO detection, and stream."""
    global cap, alert_message
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run detection with confidence filter
        weapon_results = weapon_model.predict(source=frame, save=False, save_txt=False, conf=0.7, verbose=False)
        fight_results = fight_model.predict(source=frame, save=False, save_txt=False, conf=0.7, verbose=False)

        # Extract detected objects with confidence ≥ 0.7
        detected_weapons = []
        detected_fights = []

        if weapon_results[0].boxes is not None:
            for box in weapon_results[0].boxes:
                if box.conf[0] >= 0.7:
                    class_id = int(box.cls[0])
                    class_name = weapon_model.names.get(class_id, "Unknown")
                    detected_weapons.append(class_name)

        if fight_results[0].boxes is not None:
            for box in fight_results[0].boxes:
                if box.conf[0] >= 0.7:
                    class_id = int(box.cls[0])
                    class_name = fight_model.names.get(class_id, "Unknown")
                    detected_fights.append(class_name)

        # Update alert message
        alerts = []
        if detected_weapons:
            alerts.append(f"Weapon Detected: {', '.join(detected_weapons)}")
        if detected_fights:
            alerts.append(f"Fight Detected: {', '.join(detected_fights)}")

        alert_message = " & ".join(alerts) if alerts else "No Threat Detected"

        # Draw bounding boxes manually
        frame = draw_boxes(frame, fight_results, fight_model)
        frame = draw_boxes(frame, weapon_results, weapon_model)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    """Render the index page (UI)."""
    global alert_message
    return render_template('home.html', alert_message=alert_message)

@app.route('/video_feed')
def video_feed():
    """Stream video frames."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)  # Run the Flask app
