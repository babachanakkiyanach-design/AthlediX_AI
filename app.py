import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import os
import sqlite3
from ultralytics import YOLO

# Optional: Graceful fallback if face_recognition library is not installed
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False

# =========================================================
# 1. DATABASE SETUP (SQLite for College/Athletics Roster)
# =========================================================
DB_NAME = "athletics_players.db"
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    sport TEXT,
    age INTEGER,
    height_cm REAL,
    weight_kg REAL,
    photo_path TEXT
)
''')
conn.commit()

# =========================================================
# 2. LOAD ULTRALYTICS YOLOV8 POSE MODEL
# =========================================================
@st.cache_resource
def load_ultralytics_model():
    # Downloads & caches Ultralytics YOLOv8 Nano Pose model
    return YOLO("yolov8n-pose.pt")

model = load_ultralytics_model()

# =========================================================
# 3. HELPER FUNCTIONS (Angle, Speed, Face ID)
# =========================================================
def calculate_angle(a, b, c):
    """Calculates elbow joint angle (degrees) using shoulder, elbow, and wrist."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return int(angle)

def get_known_faces():
    """Loads player face encodings from stored database photos."""
    if not FACE_REC_AVAILABLE:
        return {}
    
    cursor.execute("SELECT name, photo_path FROM players")
    rows = cursor.fetchall()
    known = {}
    for name, photo_path in rows:
        if photo_path and os.path.exists(photo_path):
            try:
                img = face_recognition.load_image_file(photo_path)
                encodings = face_recognition.face_encodings(img)
                if len(encodings) > 0:
                    known[name] = {'encoding': encodings[0]}
            except Exception:
                continue
    return known

def identify_player(frame, known_players):
    """Matches face in current video frame to registered player database."""
    if not FACE_REC_AVAILABLE or not known_players:
        return "Unknown Player"
        
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            for name, data in known_players.items():
                matches = face_recognition.compare_faces([data['encoding']], face_encoding, tolerance=0.6)
                if True in matches:
                    return name
    except Exception:
        pass
        
    return "Unknown Player"

# =========================================================
# 4. STREAMLIT WEB INTERFACE
# =========================================================
st.set_page_config(page_title="Athletics & Sports AI Hub", layout="wide", page_icon="🏃")

st.title("🏃 Athletics & Sports AI Analytics System")
st.markdown("Powered by **Ultralytics YOLOv8-Pose**. Manage player databases, identify athletes, and measure bowling/sports speed.")

tab1, tab2, tab3 = st.tabs(["📹 Video Analysis", "👤 Player Registration", "📊 Database Roster"])

# ---------------------------------------------------------
# TAB 1: VIDEO ANALYSIS & AI SPEED TRACKING
# ---------------------------------------------------------
with tab1:
    st.header("Upload Footage for Automated Analytics")
    uploaded_video = st.file_uploader("Upload Player Video (Recommended: 60 FPS)", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("🚀 Run AI Analysis"):
            with st.spinner("Analyzing motion with Ultralytics YOLOv8..."):
                known_players = get_known_faces()
                cap = cv2.VideoCapture(video_path)
                
                detected_name = "Unknown Player"
                speeds = []
                angles = []
                prev_wrist = None
                prev_time = None
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                    # Auto Face Detection on initial frames
                    if detected_name == "Unknown Player" and frame_count <= 20:
                        detected_name = identify_player(frame, known_players)

                    # Pose Tracking via Ultralytics YOLOv8
                    results = model(frame, verbose=False)
                    if results and len(results[0].keypoints) > 0:
                        kpts = results[0].keypoints.data.cpu().numpy()[0]

                        # COCO Keypoints: 6=Right Shoulder, 8=Right Elbow, 10=Right Wrist
                        if len(kpts) >= 11:
                            r_shoulder = kpts[6][:2]
                            r_elbow = kpts[8][:2]
                            r_wrist = kpts[10][:2]

                            # Calculate Elbow Joint Angle
                            if kpts[6][2] > 0.35 and kpts[8][2] > 0.35 and kpts[10][2] > 0.35:
                                angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                                angles.append(angle)

                            # Calculate Release Velocity (Distance over Delta Time)
                            if kpts[10][2] > 0.35:
                                if prev_wrist is not None and prev_time is not None:
                                    dt = curr_time - prev_time
                                    if dt > 0.005:
                                        dist_px = np.sqrt((r_wrist[0] - prev_wrist[0])**2 + (r_wrist[1] - prev_wrist[1])**2)
                                        meters_per_px = 1.7 / 380.0  # Height scale calibration factor
                                        speed_kmh = (dist_px * meters_per_px / dt) * 3.6

                                        if 15.0 < speed_kmh < 160.0:
                                            speeds.append(speed_kmh)

                                prev_wrist = r_wrist
                                prev_time = curr_time

                cap.release()

                # Display Results & Database Match
                st.subheader("🎯 Analysis Summary")

                cursor.execute("SELECT sport, age, height_cm, weight_kg FROM players WHERE name = ?", (detected_name,))
                player_profile = cursor.fetchone()

                if player_profile:
                    sport, age, height, weight = player_profile
                    st.success(f"**Identified Player:** {detected_name}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Sport", sport)
                    c2.metric("Age", f"{age} yrs")
                    c3.metric("Height", f"{height} cm")
                    c4.metric("Weight", f"{weight} kg")
                else:
                    st.info(f"**Player Identified:** {detected_name} (Not found in Database Profile)")

                top_speed = round(max(speeds), 1) if speeds else 0.0
                avg_angle = round(np.mean(angles), 1) if angles else 0.0
                score = round(min(60.0, (top_speed / 140.0) * 60.0) + (40.0 if avg_angle > 150 else (avg_angle/150.0)*40.0), 1)

                m1, m2, m3 = st.columns(3)
                m1.metric("⚡ Peak Release Speed", f"{top_speed} km/h")
                m2.metric("📐 Avg Arm Extension Angle", f"{avg_angle}°")
                m3.metric("⭐ Overall Performance Rating", f"{score} / 100")

# ---------------------------------------------------------
# TAB 2: REGISTER NEW PLAYER
# ---------------------------------------------------------
with tab2:
    st.header("Add Player to Athletics Database")
    col1, col2 = st.columns(2)

    with col1:
        p_name = st.text_input("Player Full Name (e.g., Sweety / Priya)")
        p_sport = st.selectbox("Sport Category", ["Cricket", "Football", "Athletics", "Other"])
        p_age = st.number_input("Age", min_value=10, max_value=60, value=20)

    with col2:
        p_height = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0)
        p_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=65.0)
        p_photo = st.file_uploader("Upload Clear Face Photo", type=["jpg", "jpeg", "png"])

    if st.button("Save Player to Database"):
        if p_name and p_photo:
            os.makedirs("player_photos", exist_ok=True)
            photo_path = os.path.join("player_photos", f"{p_name.lower().replace(' ', '_')}.jpg")

            with open(photo_path, "wb") as f:
                f.write(p_photo.getbuffer())

            try:
                cursor.execute(
                    "INSERT INTO players (name, sport, age, height_cm, weight_kg, photo_path) VALUES (?, ?, ?, ?, ?, ?)",
                    (p_name, p_sport, p_age, p_height, p_weight, photo_path)
                )
                conn.commit()
                st.success(f"Player '{p_name}' successfully added to database!")
            except sqlite3.IntegrityError:
                st.error("A player with this name already exists in the database.")
        else:
            st.warning("Please fill in player name and upload a face photo.")

# ---------------------------------------------------------
# TAB 3: ROSTER LEADERBOARD & DATABASE VIEW
# ---------------------------------------------------------
with tab3:
    st.header("Registered Athletics Database Roster")
    df_players = pd.read_sql_query("SELECT id, name, sport, age, height_cm, weight_kg FROM players", conn)
    st.dataframe(df_players, use_container_width=True)
    
