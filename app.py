import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import sqlite3
import mediapipe as mp
from collections import defaultdict
from datetime import datetime
from inference_sdk import InferenceHTTPClient

# Safe MediaPipe Import
try:
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_draw

# =========================================================
# 1. DATABASE SETUP
# =========================================================
DB_NAME = "athletics_players.db"
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    sport TEXT,
    position TEXT,
    age INTEGER,
    height_cm REAL,
    weight_kg REAL,
    matches_this_week INTEGER DEFAULT 0
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS performance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT,
    sport TEXT,
    position TEXT,
    timestamp TEXT,
    peak_speed REAL,
    elbow_3d_angle REAL,
    ai_fatigue INTEGER,
    readiness_score REAL,
    performance_score REAL
)
''')
conn.commit()

# =========================================================
# 2. ROBOFLOW API & MEDIAPIPE INITIALIZATION
# =========================================================
# Set your Roboflow Workspace Model ID / Workflow ID here
ROBOFLOW_MODEL_ID = "your-model-id/1"  # e.g., "cricket-bowler-tracking/1"

@st.cache_resource
def load_models():
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "YOUR_ROBOFLOW_API_KEY")
    rf_client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    pose_3d = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return rf_client, pose_3d

rf_client, mp_pose_engine = load_models()

# =========================================================
# 3. HELPER & BIOMECHANICS FUNCTIONS
# =========================================================
def calculate_3d_angle(a, b, c):
    """Calculates 3D joint angle using X, Y, Z coordinates."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return int(np.degrees(angle))

def parse_roboflow_json(rf_response):
    """
    Parses Roboflow JSON prediction response (works with standard API & Workflows).
    Extracts bounding boxes, labels, confidence, and tracker IDs.
    """
    parsed_detections = []
    
    # Handle response structure differences between standard inference & Workflows
    predictions = rf_response.get("predictions", []) if isinstance(rf_response, dict) else []
    
    if not predictions and isinstance(rf_response, list):
        predictions = rf_response

    for pred in predictions:
        if isinstance(pred, dict):
            x = pred.get("x", 0)
            y = pred.get("y", 0)
            w = pred.get("width", 0)
            h = pred.get("height", 0)
            tracker_id = pred.get("tracker_id", pred.get("detection_id", "1"))
            cls_name = pred.get("class", "person")
            confidence = pred.get("confidence", 0.0)

            # Convert center (x, y, w, h) to top-left and bottom-right corners
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            parsed_detections.append({
                "tracker_id": tracker_id,
                "class": cls_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })

    return parsed_detections

# =========================================================
# 4. STREAMLIT APP UI
# =========================================================
st.set_page_config(page_title="AthlediX AI Engine", layout="wide", page_icon="🏆")

st.title("🏆 AthlediX AI: Roboflow API + MediaPipe 3D Kinematics")
st.markdown("Powered by **Roboflow Workflow JSON API** + **Google MediaPipe 3D Pose**.")

tab1, tab2, tab3 = st.tabs(["📹 Roboflow API Analysis", "📅 Day-by-Day History", "👤 Registered Roster"])

cursor.execute("SELECT name FROM players")
registered_players = [row[0] for row in cursor.fetchall()]

BOWLER_COLORS = [(0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255)]

# ---------------------------------------------------------
# TAB 1: ROBOFLOW API MOTION ANALYSIS
# ---------------------------------------------------------
with tab1:
    st.header("Upload Video for Roboflow API Inference")
    
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_video = st.file_uploader("Upload Clip (MP4 / MOV / AVI)", type=["mp4", "mov", "avi"])
    with col_u2:
        if registered_players:
            primary_player = st.selectbox("Primary Tagged Athlete", registered_players)
        else:
            primary_player = st.text_input("Primary Athlete Name", value="Guest Bowler")
            
        model_id_input = st.text_input("Roboflow Model/Workflow ID", value=ROBOFLOW_MODEL_ID)

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        input_video_path = tfile.name

        st.subheader("Raw Input Video")
        st.video(input_video_path)

        if st.button("🚀 Run Roboflow API Pipeline"):
            with st.spinner("Processing Roboflow API predictions & 3D Pose..."):
                cap = cv2.VideoCapture(input_video_path)
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0 or np.isnan(fps):
                    fps = 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                bowler_3d_angles = defaultdict(list)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 1. API Query to Roboflow
                    try:
                        rf_result = rf_client.infer(frame, model_id=model_id_input)
                        detections = parse_roboflow_json(rf_result)
                    except Exception as e:
                        detections = []

                    # 2. Extract MediaPipe 3D Pose
                    mp_results = mp_pose_engine.process(rgb_frame)

                    angle_3d = 160
                    if mp_results.pose_world_landmarks:
                        lm = mp_results.pose_world_landmarks.landmark
                        r_shoulder, r_elbow, r_wrist = lm[12], lm[14], lm[16]
                        angle_3d = calculate_3d_angle(r_shoulder, r_elbow, r_wrist)

                    # 3. Draw Bounding Boxes from Roboflow JSON
                    for idx, det in enumerate(detections):
                        x1, y1, x2, y2 = det["bbox"]
                        t_id = det["tracker_id"]
                        cls_name = det["class"]
                        conf = det["confidence"]

                        bowler_3d_angles[t_id].append(angle_3d)

                        color = BOWLER_COLORS[idx % len(BOWLER_COLORS)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            f"ID: {t_id} | {cls_name} ({conf:.2f}) | 3D Arm: {angle_3d} deg",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

                    # Overlay header
                    cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
                    cv2.putText(frame, "AthlediX AI: Roboflow API + MediaPipe 3D Engine",
                                (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    out.write(frame)

                cap.release()
                out.release()

                st.markdown("---")
                st.subheader("🎬 AI Processed Output Video")
                st.video(output_video_path)

                # Process Metrics Leaderboard
                summary_data = []
                for b_id, angles_3d in bowler_3d_angles.items():
                    avg_a3d = np.mean(angles_3d) if angles_3d else 165.0
                    elbow_flex_3d = (max(angles_3d) - min(angles_3d)) if angles_3d else 0.0
                    icc_status = "⚠️ Non-Compliant (>15° flex)" if elbow_flex_3d > 15.0 else "✅ Legal Action (ICC Compliant)"

                    summary_data.append({
                        "Tracker ID": f"Track #{b_id}",
                        "Avg 3D Arm Angle": f"{round(avg_a3d, 1)}°",
                        "3D Flex Change": f"{elbow_flex_3d:.1f}°",
                        "ICC Arm Legality": icc_status
                    })

                st.subheader("⚡ Roboflow Tracking & Kinematics Summary")
                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: DAY-BY-DAY HISTORY
# ---------------------------------------------------------
with tab2:
    st.header("📅 Day-by-Day Performance History")
    df_filtered = pd.read_sql_query("SELECT timestamp as 'Date & Time', player_name as 'Athlete', sport as 'Sport', position as 'Position', peak_speed as 'Peak Speed (km/h)', elbow_3d_angle as '3D Arm Angle (deg)', ai_fatigue as 'AI Fatigue (1-5)', readiness_score as 'Match Readiness (%)', performance_score as 'Performance Score (100)' FROM performance_history ORDER BY timestamp DESC", conn)
    st.dataframe(df_filtered, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: REGISTER PLAYER
# ---------------------------------------------------------
with tab3:
    st.header("Add Player Profile")
    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Player Name")
        p_sport = st.selectbox("Sport Category", ["Cricket", "Football"])
        positions = ["Fast Bowler", "Spin Bowler", "Medium Pacer", "All-Rounder"] if p_sport == "Cricket" else ["Striker", "Midfielder", "Defender"]
        p_position = st.selectbox("Player Position", positions)
        p_age = st.number_input("Age", min_value=12, max_value=50, value=20)

    with col2:
        p_height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=172.0)
        p_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=140.0, value=68.0)
        p_matches = st.number_input("Matches Played This Week", min_value=0, max_value=14, value=2)

    if st.button("Save Player Profile"):
        if p_name:
            cursor.execute('''
                INSERT OR REPLACE INTO players (name, sport, position, age, height_cm, weight_kg, matches_this_week)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (p_name, p_sport, p_position, p_age, p_height, p_weight, p_matches))
            conn.commit()
            st.success(f"Player '{p_name}' profile saved!")

    st.subheader("Registered Player Roster")
    df_roster = pd.read_sql_query("SELECT id, name as 'Name', sport as 'Sport', position as 'Position', age as 'Age', height_cm as 'Height (cm)', weight_kg as 'Weight (kg)', matches_this_week as 'Matches/Wk' FROM players", conn)
    st.dataframe(df_roster, use_container_width=True)
