import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import sqlite3
import subprocess
import requests
import os

# =========================================================
# 1. MEDIAPIPE IMPORT
# =========================================================
import mediapipe as mp

try:
    import mediapipe.python.solutions.pose as mp_pose
except AttributeError:
    try:
        mp_pose = mp.solutions.pose
    except Exception:
        mp_pose = None

# =========================================================
# 2. DATABASE SETUP
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
conn.commit()

# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def calculate_3d_angle(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return int(np.degrees(angle))

def query_roboflow_api(frame, api_key, project_id, version=1):
    """Sends a frame directly to Roboflow Inference API."""
    try:
        _, encoded_img = cv2.imencode(".jpg", frame)
        img_bytes = encoded_img.tobytes()

        url = f"https://detect.roboflow.com/{project_id}/{version}?api_key={api_key}"
        response = requests.post(
            url,
            data=img_bytes,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        if response.status_code == 200:
            return response.json().get("predictions", [])
        else:
            st.warning(f"Roboflow API returned status code {response.status_code}: {response.text}")
            return []
    except Exception as e:
        st.error(f"Roboflow Connection Error: {e}")
        return []

def convert_to_h264(input_path, output_path):
    try:
        command = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vcodec', 'libx264',
            '-crf', '23',
            '-preset', 'fast',
            output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except Exception:
        return input_path

# =========================================================
# 4. STREAMLIT APP UI
# =========================================================
st.set_page_config(page_title="AthlediX AI Engine", layout="wide", page_icon="🏆")
st.title("🏆 AthlediX AI: Cricket Detection & Pose Engine")

# Configuration Sidebar
st.sidebar.header("🔑 Roboflow Credentials")
rf_api_key = st.sidebar.text_input("Roboflow API Key", value=st.secrets.get("ROBOFLOW_API_KEY", ""), type="password")
rf_project_id = st.sidebar.text_input("Project ID", value="avs-cricket-player-and-ball-detection")
rf_version = st.sidebar.number_input("Model Version", min_value=1, max_value=20, value=1)
frame_skip = st.sidebar.slider("Frame Skip Optimization", min_value=1, max_value=10, value=3)

tab1, tab2 = st.tabs(["📹 Detection Analysis", "👤 Roster"])

with tab1:
    uploaded_video = st.file_uploader("Upload Video (MP4 / MOV / AVI)", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        input_video_path = tfile.name

        st.subheader("Raw Video")
        st.video(input_video_path)

        if st.button("🚀 Process Video with Roboflow"):
            if not rf_api_key:
                st.error("Please enter your Roboflow API Key in the sidebar or Streamlit Secrets!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                cap = cv2.VideoCapture(input_video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0 or np.isnan(fps):
                    fps = 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                temp_raw_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_raw_video, fourcc, fps, (width, height))

                pose_engine = mp_pose.Pose(min_detection_confidence=0.5) if mp_pose else None

                frame_idx = 0
                last_predictions = []
                angle_3d = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1

                    # Run inference every N frames
                    if frame_idx % frame_skip == 0 or frame_idx == 1:
                        last_predictions = query_roboflow_api(frame, rf_api_key, rf_project_id, rf_version)
                        
                        if pose_engine:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose_engine.process(rgb_frame)
                            if results.pose_world_landmarks:
                                lm = results.pose_world_landmarks.landmark
                                angle_3d = calculate_3d_angle(lm[12], lm[14], lm[16])

                    # Draw Bounding Boxes from Roboflow
                    for pred in last_predictions:
                        x, y = pred.get("x", 0), pred.get("y", 0)
                        w, h = pred.get("width", 0), pred.get("height", 0)
                        cls_name = pred.get("class", "object")
                        conf = pred.get("confidence", 0.0)

                        x1, y1 = int(x - w / 2), int(y - h / 2)
                        x2, y2 = int(x + w / 2), int(y + h / 2)

                        # Color based on detection type
                        color = (0, 255, 0) if "ball" in cls_name.lower() else (255, 165, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{cls_name} ({conf:.2f})", (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw Arm Angle Overlay
                    if angle_3d > 0:
                        cv2.putText(frame, f"3D Arm Angle: {angle_3d} deg", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    out.write(frame)

                    if total_frames > 0:
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))
                        status_text.text(f"Processing frame {frame_idx} / {total_frames}...")

                cap.release()
                out.release()
                status_text.text("Encoding final video format...")

                final_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                convert_to_h264(temp_raw_video, final_video)

                status_text.empty()
                progress_bar.empty()
                st.success("Processing complete!")

                with open(final_video, 'rb') as vf:
                    st.video(vf.read())

with tab2:
    st.header("Registered Player Roster")
    df_roster = pd.read_sql_query("SELECT id, name as 'Name', sport as 'Sport', position as 'Position' FROM players", conn)
    st.dataframe(df_roster, use_container_width=True)
