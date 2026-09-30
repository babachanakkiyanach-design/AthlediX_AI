import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import subprocess
from ultralytics import YOLO

# =========================================================
# 1. STREAMLIT APP CONFIGURATION
# =========================================================
st.set_page_config(page_title="AthlediX AI: YOLO11 Performance Engine", layout="wide", page_icon="🏆")
st.title("🏆 AthlediX AI: Player Detection, Speed & Performance Tracker")

# Load YOLO11 Model with Object Tracking capabilities
@st.cache_resource
def load_yolo_model():
    return YOLO("yolo11n.pt")  # Nano model for fast cloud execution

yolo_model = load_yolo_model()

# Session State for Player Tracking History
if "player_history" not in st.session_state:
    st.session_state.player_history = {}

# Sidebar Controls
st.sidebar.header("⚙️ Detection & Speed Settings")
conf_thresh = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
frame_skip = st.sidebar.slider("Frame Skip (Speed Optimization)", 1, 10, 2)
pixel_to_meter = st.sidebar.number_input("Pixel to Meter Conversion Factor", value=0.05, step=0.01)

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def convert_to_h264(input_path, output_path):
    """Encodes raw video output for browser compatibility using system ffmpeg."""
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

def compute_speed_kmh(p1, p2, time_interval, p2m_factor):
    """Calculates speed in km/h based on pixel displacement."""
    pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    meters_distance = pixel_distance * p2m_factor
    speed_m_per_s = meters_distance / time_interval if time_interval > 0 else 0
    return speed_m_per_s * 3.6  # Convert m/s to km/h

# =========================================================
# 3. INTERFACE TABS
# =========================================================
tab1, tab2 = st.tabs(["📹 Video Processing & Speed Analytics", "📊 Player History & Performance"])

with tab1:
    uploaded_video = st.file_uploader("Upload Match Video (MP4 / MOV / AVI)", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        input_path = tfile.name

        st.subheader("Uploaded Input Video")
        st.video(input_path)

        if st.button("🚀 Track Players & Calculate Speeds"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(temp_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            # Positional Tracking Buffers
            previous_positions = {}  # {track_id: (x, y)}
            frame_idx = 0
            time_per_processed_frame = (1 / fps) * frame_skip

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Track objects using YOLO11 ByteTrack/BoT-SORT
                if frame_idx % frame_skip == 0 or frame_idx == 1:
                    results = yolo_model.track(frame, conf=conf_thresh, persist=True, verbose=False, classes=[0])[0]

                    if results.boxes is not None and results.boxes.id is not None:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        track_ids = results.boxes.id.int().cpu().numpy()

                        for box, track_id in zip(boxes, track_ids):
                            x1, y1, x2, y2 = map(int, box)
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            current_pos = (center_x, center_y)

                            # Speed Calculation
                            speed_kmh = 0.0
                            if track_id in previous_positions:
                                prev_pos = previous_positions[track_id]
                                speed_kmh = compute_speed_kmh(prev_pos, current_pos, time_per_processed_frame, pixel_to_meter)

                            previous_positions[track_id] = current_pos

                            # Record Performance History in Session State
                            player_key = f"Player #{track_id}"
                            if player_key not in st.session_state.player_history:
                                st.session_state.player_history[player_key] = {
                                    "max_speed": round(speed_kmh, 1),
                                    "speed_records": [speed_kmh],
                                    "detections": 1
                                }
                            else:
                                hist = st.session_state.player_history[player_key]
                                hist["speed_records"].append(speed_kmh)
                                hist["max_speed"] = max(hist["max_speed"], round(speed_kmh, 1))
                                hist["detections"] += 1

                            # Draw Bounding Box & Speed Label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID:{track_id} | {speed_kmh:.1f} km/h", (x1, max(20, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                out.write(frame)

                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    status_text.text(f"Analyzing frame {frame_idx} / {total_frames}...")

            cap.release()
            out.release()

            status_text.text("Encoding final video format...")
            final_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            convert_to_h264(temp_raw, final_video)

            status_text.empty()
            progress_bar.empty()
            st.success("Analysis Complete!")

            with open(final_video, 'rb') as vf:
                st.video(vf.read())

with tab2:
    st.header("📊 Player Performance & History Logs")

    if st.session_state.player_history:
        summary_data = []

        for player_id, data in st.session_state.player_history.items():
            avg_speed = np.mean(data["speed_records"]) if data["speed_records"] else 0
            summary_data.append({
                "Player ID": player_id,
                "Max Speed (km/h)": data["max_speed"],
                "Average Speed (km/h)": round(avg_speed, 1),
                "Total Detection Frames": data["detections"],
                "Performance Rank": "High Workrate" if avg_speed > 10 else "Moderate Workrate"
            })

        df_summary = pd.DataFrame(summary_data)
        
        # Display Key Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Players Tracked", len(df_summary))
        col2.metric("Top Speed Recorded", f"{df_summary['Max Speed (km/h)'].max()} km/h")
        col3.metric("Average Match Speed", f"{round(df_summary['Average Speed (km/h)'].mean(), 1)} km/h")

        st.subheader("Detailed Performance Roster")
        st.dataframe(df_summary, use_container_width=True)

        if st.button("🧹 Clear Player History"):
            st.session_state.player_history = {}
            st.rerun()
    else:
        st.info("No player history available yet. Upload and process a video in Tab 1 first!")
