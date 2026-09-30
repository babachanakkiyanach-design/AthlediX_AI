import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import sqlite3
from collections import defaultdict
from inference_sdk import InferenceHTTPClient

# =========================================================
# 1. SAFE MEDIAPIPE IMPORTS (Fixes ModuleNotFoundError)
# =========================================================
import mediapipe as mp

try:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_draw
except ImportError:
    try:
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
    except AttributeError:
        mp_pose = None
        mp_draw = None

# =========================================================
# 2. DATABASE INITIALIZATION
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
# 3. ROBOFLOW WORKFLOW & MEDIAPIPE SETUP
# =========================================================
WORKSPACE_NAME = "baba-chanakkiyanach"
WORKFLOW_ID = "avs-cricket-player-and-ball-detection"

@st.cache_resource
def load_models():
    # Load Roboflow API key from secrets or fallback to key
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "QtbCoMHIHOWY121P7mvP")
    
    rf_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    pose_3d = None
    if mp_pose is not None:
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
# 4. HELPER FUNCTIONS
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

def process_roboflow_workflow_frame(frame):
    """Encodes frame and queries Roboflow Workflow API."""
    try:
        # Encode frame to JPG memory buffer for workflow API transmission
        _, encoded_img = cv2.imencode(".jpg", frame)
        img_bytes = encoded_img.tobytes()

        response = rf_client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": img_bytes}
        )
        return response
    except Exception as e:
        return None

# =========================================================
# 5. STREAMLIT INTERFACE
# =========================================================
st.set_page_config(page_title="AthlediX AI Engine", layout="wide", page_icon="🏆")
st.title("🏆 AthlediX AI: Roboflow Workflow + MediaPipe Kinematics")

tab1, tab2, tab3 = st.tabs(["📹 Workflow Analysis", "📅 Day-by-Day History", "👤 Registered Roster"])

cursor.execute("SELECT name FROM players")
registered_players = [row[0] for row in cursor.fetchall()]

BOWLER_COLORS = [(0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255)]

with tab1:
    st.header("Upload Video for Roboflow Workflow Inference")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_video = st.file_uploader("Upload Video (MP4 / MOV / AVI)", type=["mp4", "mov", "avi"])
    with col2:
        primary_player = st.selectbox("Athlete Profile", registered_players) if registered_players else st.text_input("Athlete Name", value="Player 1")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        input_video_path = tfile.name

        st.subheader("Raw Input Video")
        st.video(input_video_path)

        if st.button("🚀 Run Roboflow Workflow"):
            with st.spinner("Executing Roboflow Serverless Workflow & MediaPipe..."):
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

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 1. Query Roboflow Serverless Workflow
                    workflow_output = process_roboflow_workflow_frame(frame)

                    # 2. Calculate MediaPipe Pose Angle
                    angle_3d = 160
                    if mp_pose_engine is not None:
                        mp_results = mp_pose_engine.process(rgb_frame)
                        if mp_results.pose_world_landmarks:
                            lm = mp_results.pose_world_landmarks.landmark
                            angle_3d = calculate_3d_angle(lm[12], lm[14], lm[16])

                    # 3. Parse and Draw Workflow Output Predictions
                    if workflow_output and isinstance(workflow_output, list) and len(workflow_output) > 0:
                        predictions = workflow_output[0].get("predictions", [])
                        for idx, pred in enumerate(predictions):
                            if isinstance(pred, dict):
                                x, y = pred.get("x", 0), pred.get("y", 0)
                                w, h = pred.get("width", 0), pred.get("height", 0)
                                t_id = pred.get("tracker_id", pred.get("detection_id", "1"))
                                cls_name = pred.get("class", "object")
                                conf = pred.get("confidence", 0.0)

                                x1, y1 = int(x - w / 2), int(y - h / 2)
                                x2, y2 = int(x + w / 2), int(y + h / 2)

                                bowler_3d_angles[t_id].append(angle_3d)

                                color = BOWLER_COLORS[idx % len(BOWLER_COLORS)]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(
                                    frame,
                                    f"ID:{t_id} {cls_name} ({conf:.2f}) | Arm: {angle_3d}deg",
                                    (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                                )

                    # Overlay Title
                    cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
                    cv2.putText(frame, "AthlediX AI: Roboflow Workflow + MediaPipe 3D Engine",
                                (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    out.write(frame)

                cap.release()
                out.release()

                st.subheader("🎬 AI Processed Output Video")
                st.video(output_video_path)

# ---------------------------------------------------------
# TAB 2 & TAB 3: HISTORY & ROSTER
# ---------------------------------------------------------
with tab2:
    st.header("📅 Day-by-Day Performance History")
    df_filtered = pd.read_sql_query("SELECT timestamp as 'Date & Time', player_name as 'Athlete', sport as 'Sport', position as 'Position', peak_speed as 'Peak Speed (km/h)', elbow_3d_angle as '3D Arm Angle (deg)', ai_fatigue as 'AI Fatigue (1-5)', readiness_score as 'Match Readiness (%)', performance_score as 'Performance Score (100)' FROM performance_history ORDER BY timestamp DESC", conn)
    st.dataframe(df_filtered, use_container_width=True)

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
