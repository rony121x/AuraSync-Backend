import base64
import numpy as np
import math
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from deepface import DeepFace
import cv2
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# ------------------ ENVIRONMENT CONFIG ------------------
# Load local environment variables (if running on your laptop)
load_dotenv()

app = Flask(__name__)

# Dynamically set the allowed frontend URL
# If on Render, it uses the Vercel link. If on your laptop, it defaults to localhost.
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
CORS(app, resources={r"/api/*": {"origins": FRONTEND_URL}}, supports_credentials=True)

# ------------------ DATABASE ------------------
# Dynamically connect to the Database
# If on Render, it uses Atlas. If on your laptop, it defaults to local MongoDB.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["smart_attendance"]

MODEL = "Facenet"
THRESHOLD = 0.7

# ------------------ SECURITY CONFIG ------------------
TEACHER_ACCESS_CODE = "AURA-ADMIN-2026"

# Geofencing (Adjust to your testing coordinates)
CAMPUS_LAT = 19.0760
CAMPUS_LON = 72.8777
MAX_RADIUS_KM = 10 

ALLOWED_IPS = ["127.0.0.1", "::1"] 
ENABLE_IP_CHECK = False 

# ------------------ UTILS ------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def process_base64_image(base64_str):
    try:
        if "," in base64_str: base64_str = base64_str.split(",", 1)[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(base64_str), np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(img, (640, 640)) if img is not None else None
    except Exception: return None

def get_face_embedding(img):
    try:
        return DeepFace.represent(img_path=img, model_name=MODEL, enforce_detection=False, detector_backend="opencv")[0]["embedding"]
    except Exception: return None

# ------------------ AUTH ROUTES ------------------
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.json
    name, email, password = data.get("name"), data.get("email"), data.get("password")
    role, image, subject = data.get("role", "student"), data.get("image"), data.get("subject")
    admin_code = data.get("admin_code")

    if not all([name, email, password, image]):
        return jsonify({"error": "All fields are required"}), 400

    if role == "teacher":
        if not subject: return jsonify({"error": "Teachers must specify a subject"}), 400
        if admin_code != TEACHER_ACCESS_CODE: return jsonify({"error": "Invalid Teacher Access Code."}), 403

    if db.users.find_one({"email": email}):
        return jsonify({"error": "Email already in use"}), 409

    img = process_base64_image(image)
    embedding = get_face_embedding(img) if img is not None else None

    if embedding is None: return jsonify({"error": "Could not detect a clear face."}), 400

    db.users.insert_one({
        "name": name, "email": email, "password": generate_password_hash(password),
        "role": role, "subject": subject if role == "teacher" else None,
        "embeddings": [embedding], "created_at": datetime.now()
    })
    return jsonify({"message": "Registration successful"}), 201

@app.route("/api/auth/scan-face", methods=["POST"])
def scan_face():
    img = process_base64_image(request.json.get("image"))
    new_embedding = get_face_embedding(img) if img is not None else None
    if new_embedding is None: return jsonify({"error": "No face detected"}), 400

    for user in db.users.find():
        for emb in user.get("embeddings", []):
            if np.dot(emb, new_embedding) / (np.linalg.norm(emb) * np.linalg.norm(new_embedding)) > THRESHOLD:
                return jsonify({"name": user["name"], "email": user["email"], "role": user["role"]})
    return jsonify({"error": "Face not recognized"}), 404

@app.route("/api/auth/verify-password", methods=["POST"])
def verify_password():
    data = request.json
    email, password = data.get("email"), data.get("password")
    user_lat, user_lon = data.get("lat"), data.get("lon")

    if ENABLE_IP_CHECK and request.remote_addr not in ALLOWED_IPS:
        return jsonify({"error": "Login denied: Unauthorized IP address."}), 403

    if not user_lat or not user_lon:
        return jsonify({"error": "Location access is required to log in."}), 400
    
    distance = haversine(CAMPUS_LAT, CAMPUS_LON, float(user_lat), float(user_lon))
    if distance > MAX_RADIUS_KM:
        return jsonify({"error": f"Login denied: You are {round(distance, 1)}km away. Must be within {MAX_RADIUS_KM}km of campus."}), 403

    user = db.users.find_one({"email": email})
    if user and check_password_hash(user["password"], password):
        return jsonify({"message": "Success", "user": {"name": user["name"], "email": user["email"], "role": user["role"], "subject": user.get("subject")}})
    
    return jsonify({"error": "Invalid credentials"}), 401

# ------------------ ATTENDANCE & ANALYTICS ROUTES ------------------
@app.route("/api/attendance/session", methods=["GET", "POST"])
def manage_session():
    if request.method == "POST":
        data = request.json
        email, is_active = data.get("email"), data.get("is_active")
        teacher = db.users.find_one({"email": email, "role": "teacher"})
        if not teacher: return jsonify({"error": "Unauthorized"}), 403
        
        subject = teacher.get("subject")
        
        # 1. Update the live status
        db.active_sessions.update_one({"subject": subject}, {"$set": {"is_active": is_active, "teacher": email}}, upsert=True)
        
        # 2. LOG THE SESSION (This increments the "Sessions" counter on the dashboard)
        if is_active:
            today = datetime.now().strftime("%Y-%m-%d")
            # Using $setOnInsert ensures we only count 1 session per day per subject, 
            # even if the teacher toggles it on and off multiple times in a single day.
            db.session_history.update_one(
                {"subject": subject, "date": today},
                {"$setOnInsert": {"opened_at": datetime.now(), "teacher": email}},
                upsert=True
            )

        return jsonify({"message": f"{subject} attendance is now {'OPENED' if is_active else 'CLOSED'}."})
    else:
        active = list(db.active_sessions.find({"is_active": True}, {"_id": 0}))
        return jsonify([s["subject"] for s in active])

@app.route("/api/attendance/mark", methods=["POST"])
def mark_attendance():
    data = request.json
    email, subject = data.get("email"), data.get("subject")
    
    if not db.active_sessions.find_one({"subject": subject, "is_active": True}):
        return jsonify({"error": f"{subject} is currently closed."}), 400
        
    today = datetime.now().strftime("%Y-%m-%d")
    if db.attendance.find_one({"email": email, "subject": subject, "date": today}):
        return jsonify({"error": f"Already marked present for {subject} today."}), 400
        
    user = db.users.find_one({"email": email})
    db.attendance.insert_one({
        "email": email, "name": user["name"], "subject": subject,
        "timestamp": datetime.now(), "date": today
    })
    return jsonify({"message": f"Successfully marked present for {subject}!"})

@app.route("/api/analytics", methods=["GET"])
def analytics():
    role, email = request.args.get("role"), request.args.get("email")

    if role == "student":
        records = list(db.attendance.aggregate([
            {"$match": {"email": email}},
            {"$group": {"_id": "$subject", "present": {"$sum": 1}}}
        ]))
        subjects_data = []
        for rec in records:
            subj = rec["_id"]
            
            # Fetch actual teacher sessions from the new session_history table
            total_days = db.session_history.count_documents({"subject": subj})
            math_days = max(total_days, 1) # Prevent division by zero
            
            subjects_data.append({
                "subject": subj, "percentage": round((rec["present"] / math_days) * 100, 1),
                "present": rec["present"], "total": total_days
            })
        return jsonify({"subjects": subjects_data})

    elif role == "teacher":
        teacher = db.users.find_one({"email": email})
        subject = teacher.get("subject")
        
        # Count total unique days the teacher has clicked "Open Attendance"
        total_days = db.session_history.count_documents({"subject": subject})
        math_days = max(total_days, 1) # Prevent division by zero for the math below
        
        daily = list(db.attendance.aggregate([
            {"$match": {"subject": subject}},
            {"$group": {"_id": "$date", "students": {"$sum": 1}}}, 
            {"$sort": {"_id": 1}}
        ]))

        enrolled_students = list(db.users.find({"role": "student"}))
        safe_count, risk_count = 0, 0
        for st in enrolled_students:
            present_days = db.attendance.count_documents({"email": st["email"], "subject": subject})
            if (present_days / math_days) * 100 >= 75: safe_count += 1
            else: risk_count += 1

        session = db.active_sessions.find_one({"subject": subject})
        
        return jsonify({
            "subject": subject,
            "is_active": session.get("is_active", False) if session else False,
            "total_students": len(enrolled_students),
            "trends": [{"date": d["_id"], "students": d["students"]} for d in daily],
            "risk_data": [{"name": "Safe (≥75%)", "value": safe_count}, {"name": "At Risk (<75%)", "value": risk_count}],
            "total_days": total_days # Returns the precise session count to the frontend
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
