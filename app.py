import os
import cv2
import json
import datetime
from flask import Flask, render_template, Response, jsonify, redirect, url_for, request, send_from_directory, session
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load YOLO models
weapon_model = YOLO(r"C:/Users/LENOVO/Downloads/weapon.pt")
fight_model = YOLO(r"C:/Users/LENOVO/Downloads/fight.pt")

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
DATABASE_FILE = "database.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

alert_message = "No threats detected"

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def load_data():
    """Load data from the JSON database."""
    if not os.path.exists(DATABASE_FILE):
        return {"users": [], "threat_records": []}
    with open(DATABASE_FILE, "r") as file:
        return json.load(file)

def save_data(data):
    """Save data to the JSON database."""
    with open(DATABASE_FILE, "w") as file:
        json.dump(data, file, indent=4)

def save_threat(threat_class):
    """Save detected threats with timestamp to the database."""
    data = load_data()
    data["threat_records"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "class": threat_class
    })
    save_data(data)

def detect_threats(frame):
    """Detect threats from both models and annotate the frame."""
    weapon_results = weapon_model(frame)
    fight_results = fight_model(frame)

    detected_threats = []
    for results, model in [(weapon_results, weapon_model), (fight_results, fight_model)]:
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    color = (0, 255, 0) if "gun" in class_name or "knife" in class_name else (0, 0, 255)
                    label = f"{class_name}: {float(box.conf[0]):.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    detected_threats.append(class_name)
                    save_threat(class_name)
    return frame, detected_threats

def generate_frames():
    """Video streaming function with threat detection."""
    global alert_message
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_threats = detect_threats(frame)
        alert_message = f"Threat Detected: {', '.join(detected_threats)}" if detected_threats else "No threats detected"
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("home.html", alert_message=alert_message)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_alert")
def get_alert():
    return jsonify({"alert": alert_message})

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        data = load_data()
        for user in data["users"]:
            if user["email"] == email and user["password"] == password:
                session['user'] = user
                return redirect(url_for('home'))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        data = load_data()
        if any(user["email"] == email for user in data["users"]):
            return render_template("register.html", error="Email already exists")
        data["users"].append({"name": name, "email": email, "password": password})
        save_data(data)
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload and perform threat detection."""
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400
    
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    
    image = cv2.imread(save_path)
    image, detected_threats = detect_threats(image)
    processed_path = os.path.join(PROCESSED_FOLDER, file.filename)
    cv2.imwrite(processed_path, image)
    os.remove(save_path)
    
    alert_message = f"Threat Detected: {', '.join(detected_threats)}" if detected_threats else "No threats detected"
    return jsonify({"message": alert_message, "image_url": f"/processed/{file.filename}"})

@app.route("/processed/<filename>")
def processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    finally:
        cap.release()
        cv2.destroyAllWindows()
