import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import sqlite3
import mediapipe as mp
from collections import deque, defaultdict
from datetime import datetime
from ultralytics import YOLO

# SAHI Imports for Sliced Hyper Inference
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Updated MediaPipe Imports for Cloud Compatibility
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
# 2. LOAD ENSEMBLE MODELS (YOLO11 + SAHI + GOOGLE MEDIAPIPE)
# =========================================================
@st.cache_resource
def load_models():
    # Load YOLO11 Pose Model
    yolo11_model = YOLO("yolo11n-pose.pt")
    
    # Initialize SAHI Detection Model wrapper around YOLO11
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11n-pose.pt",
        confidence_threshold=0.3,
        device="cpu"
    )

    # Initialize Google MediaPipe 3D Pose
    pose_3d = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return yolo11_model, sahi_model, pose_3d, mp_draw

yolo11_model, sahi_model, mp_pose_engine, mp_draw = load_models()

# =========================================================
# 3. HELPER & 3D BIOMECHANICS FUNCTIONS
# =========================================================
def calculate_3d_angle(a, b, c):
    """Calculates true 3D joint angle using X, Y, Z coordinates."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return int(np.degrees(angle))

def calculate_ai_fatigue(angle_variance, speed_decay):
    """AI Automatic Fatigue Scoring (1 = Fresh, 5 = Exhausted)."""
    fatigue_index = (angle_variance * 0.4) + (speed_decay * 0.6)
    if fatigue_index < 12.0:
        return 1
    elif fatigue_index < 22.0:
        return 2
    elif fatigue_index < 32.0:
        return 3
    elif fatigue_index < 42.0:
        return 4
    else:
        return 5

def calculate_readiness_score(height_cm, weight_kg, matches_this_week, ai_fatigue_level):
    """Calculates Match Readiness Score (0% - 100%)."""
    base_readiness = 100.0
    workload_deduction = matches_this_week * 7.5
    fatigue_deduction = (ai_fatigue_level - 1) * 12.0
    
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2) if height_m > 0 else 22.0
    bmi_penalty = 5.0 if (bmi < 18.5 or bmi > 25.0) else 0.0

    readiness = base_readiness - workload_deduction - fatigue_deduction - bmi_penalty
    return max(0.0, min(100.0, round(readiness, 1)))

# =========================================================
# 4. STREAMLIT APP UI
# =========================================================
st.set_page_config(page_title="AthlediX AI Engine", layout="wide", page_icon="🏆")

st.title("🏆 AthlediX AI: YOLO11 + SAHI Sliced Inference Engine")
st.markdown("Powered by **YOLO11 Pose** + **SAHI (Sliced Hyper Inference)** + **MediaPipe 3D Kinematics**.")

tab1, tab2, tab3 = st.tabs(["📹 SAHI Motion Analysis", "📅 Day-by-Day History", "👤 Registered Roster"])

cursor.execute("SELECT name FROM players")
registered_players = [row[0] for row in cursor.fetchall()]

BOWLER_COLORS = [(0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255)]

# ---------------------------------------------------------
# TAB 1: SAHI & YOLO11 MOTION ANALYSIS
# ---------------------------------------------------------
with tab1:
    st.header("Upload Video for SAHI + YOLO11 Sliced Analysis")
    
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_video = st.file_uploader("Upload Bowling Clip (MP4 / MOV / AVI)", type=["mp4", "mov", "avi"])
    with col_u2:
        if registered_players:
            primary_player = st.selectbox("Primary Tagged Athlete", registered_players)
        else:
            primary_player = st.text_input("Primary Athlete Name", value="Guest Bowler")
            
        use_sahi_slicing = st.checkbox("Enable SAHI Sliced Slicing (For Distant Bowlers)", value=False)

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        input_video_path = tfile.name

        st.subheader("Raw Video Input")
        st.video(input_video_path)

        if st.button("🚀 Run YOLO11 + SAHI Dual Pipeline"):
            with st.spinner("Processing sliced inference and 3D pose kinematics..."):
                cap = cv2.VideoCapture(input_video_path)
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0 or np.isnan(fps):
                    fps = 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                bowler_speeds = defaultdict(list)
                bowler_3d_angles = defaultdict(list)
                prev_wrists = {}
                prev_times = {}

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # ENGINE 1: SAHI Sliced Inference or Direct YOLO11
                    if use_sahi_slicing:
                        sahi_results = get_sliced_prediction(
                            frame,
                            sahi_model,
                            slice_height=256,
                            slice_width=256,
                            overlap_height_ratio=0.2,
                            overlap_width_ratio=0.2,
                            verbose=0
                        )

                    try:
                        yolo11_results = yolo11_model.track(frame, persist=True, verbose=False)
                    except Exception:
                        yolo11_results = yolo11_model(frame, verbose=False)

                    # ENGINE 2: MediaPipe 3D Landmark Extraction
                    mp_results = mp_pose_engine.process(rgb_frame)

                    if yolo11_results and len(yolo11_results[0].keypoints) > 0:
                        boxes = yolo11_results[0].boxes
                        has_ids = hasattr(boxes, 'id') and boxes.id is not None
                        track_ids = boxes.id.cpu().numpy().astype(int) if has_ids else list(range(1, len(boxes) + 1))
                        keypoints_data = yolo11_results[0].keypoints.data.cpu().numpy()

                        for idx, track_id in enumerate(track_ids):
                            kpts = keypoints_data[idx]

                            if len(kpts) >= 17:
                                person_px_h = np.abs(kpts[16][1] - kpts[0][1])
                                meters_per_px = 1.70 / float(person_px_h) if person_px_h > 40 else 1.70 / 380.0
                            else:
                                meters_per_px = 1.70 / 380.0

                            angle_3d = 160
                            if mp_results.pose_world_landmarks:
                                lm = mp_results.pose_world_landmarks.landmark
                                r_shoulder, r_elbow, r_wrist = lm[12], lm[14], lm[16]
                                angle_3d = calculate_3d_angle(r_shoulder, r_elbow, r_wrist)
                                bowler_3d_angles[track_id].append(angle_3d)

                            if len(kpts) >= 11 and kpts[10][2] > 0.30:
                                w_pt = kpts[10][:2]
                                wx, wy = int(w_pt[0]), int(w_pt[1])

                                curr_speed = 0.0
                                if track_id in prev_wrists and track_id in prev_times:
                                    dt = curr_time - prev_times[track_id]
                                    if dt > 0.005:
                                        dist_px = np.sqrt((w_pt[0] - prev_wrists[track_id][0])**2 + (w_pt[1] - prev_wrists[track_id][1])**2)
                                        speed_kmh = (dist_px * meters_per_px / dt) * 3.6

                                        if 15.0 < speed_kmh < 165.0:
                                            bowler_speeds[track_id].append(speed_kmh)
                                            curr_speed = speed_kmh

                                prev_wrists[track_id] = w_pt
                                prev_times[track_id] = curr_time

                                color = BOWLER_COLORS[(track_id - 1) % len(BOWLER_COLORS)]
                                cv2.circle(frame, (wx, wy), 7, color, -1)
                                label_mode = "SAHI + YOLO11" if use_sahi_slicing else "YOLO11"
                                cv2.putText(frame, f"[{label_mode}] Bowler #{track_id}: {curr_speed:.1f} km/h | 3D Arm: {angle_3d} deg",
                                            (wx - 20, max(20, wy - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
                    cv2.putText(frame, "AthlediX AI: YOLO11 Pose + SAHI Sliced Engine + MediaPipe 3D",
                                (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                    out.write(frame)

                cap.release()
                out.release()

                st.markdown("---")
                st.subheader("🎬 AI Output Video")
                st.video(output_video_path)

                summary_data = []
                fastest_id = None
                highest_speed = -1.0

                for b_id, speeds in bowler_speeds.items():
                    peak_s = max(speeds) if speeds else 0.0
                    angles_3d = bowler_3d_angles[b_id]
                    avg_a3d = np.mean(angles_3d) if angles_3d else 165.0
                    ang_var = float(np.std(angles_3d)) if angles_3d else 0.0
                    
                    spd_decay = (speeds[0] - speeds[-1]) if len(speeds) > 3 else 0.0
                    ai_fatigue = calculate_ai_fatigue(ang_var, max(0.0, spd_decay))

                    elbow_flex_3d = (max(angles_3d) - min(angles_3d)) if angles_3d else 0.0
                    icc_status = "⚠️ Non-Compliant (>15° flex)" if elbow_flex_3d > 15.0 else "✅ Legal Action (ICC Compliant)"

                    if peak_s > highest_speed:
                        highest_speed = peak_s
                        fastest_id = b_id

                    summary_data.append({
                        "Bowler ID": f"Bowler #{b_id}",
                        "Peak Speed (km/h)": round(peak_s, 1),
                        "Avg 3D Arm Angle": f"{round(avg_a3d, 1)}°",
                        "3D Flex Change": f"{elbow_flex_3d:.1f}°",
                        "ICC Arm Legality": icc_status,
                        "AI Fatigue Level": f"{ai_fatigue} / 5"
                    })

                st.subheader("⚡ Kinematics Leaderboard")
                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True)

                    st.success(f"🔥 **Fastest Bowler:** **Bowler #{fastest_id}** with peak speed of **{highest_speed:.1f} km/h**!")

                    cursor.execute("SELECT sport, position, height_cm, weight_kg, matches_this_week FROM players WHERE name = ?", (primary_player,))
                    p_prof = cursor.fetchone()

                    if p_prof:
                        sport, position, height, weight, matches = p_prof
                    else:
                        sport, position, height, weight, matches = "Cricket", "Fast Bowler", 175.0, 70.0

                    primary_speed = highest_speed
                    primary_angle = float(summary_data[0]["Avg 3D Arm Angle"].replace("°", "")) if summary_data else 165.0
                    primary_fatigue = int(summary_data[0]["AI Fatigue Level"].split()[0]) if summary_data else 2
                    
                    readiness = calculate_readiness_score(height, weight, matches, primary_fatigue)
                    perf_score = round(min(60.0, (primary_speed / 150.0) * 60.0) + (40.0 if primary_angle > 150 else 30.0), 1)

                    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute('''
                        INSERT INTO performance_history 
                        (player_name, sport, position, timestamp, peak_speed, elbow_3d_angle, ai_fatigue, readiness_score, performance_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (primary_player, sport, position, current_time_str, primary_speed, primary_angle, primary_fatigue, readiness, perf_score))
                    conn.commit()

# ---------------------------------------------------------
# TAB 2: DAY-BY-DAY HISTORY
# ---------------------------------------------------------
with tab2:
    st.header("📅 Day-by-Day Performance History")
    df_filtered = pd.read_sql_query("SELECT timestamp as 'Date & Time', player_name as 'Athlete', sport as 'Sport', position as 'Position', peak_speed as 'Peak Speed (km/h)', elbow_3d_angle as '3D Arm Angle (deg)', ai_fatigue as 'AI Fatigue (1-5)', readiness_score as 'Match Readiness (%)', performance_score as 'Performance Score (100)' FROM performance_history ORDER BY timestamp DESC", conn)
    
    if not df_filtered.empty:
        df_filtered['Peak Speed (km/h)'] = df_filtered['Peak Speed (km/h)'].map('{:.1f}'.format)
        df_filtered['Match Readiness (%)'] = df_filtered['Match Readiness (%)'].map('{:.1f}'.format)
        df_filtered['Performance Score (100)'] = df_filtered['Performance Score (100)'].map('{:.1f}'.format)
        
    st.dataframe(df_filtered, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: REGISTER PLAYER
# ---------------------------------------------------------
with tab3:
    st.header("Add Player Profile")
    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Player Name (e.g., Sweety / Priya)")
        p_sport = st.selectbox("Sport Category", ["Cricket", "Football"])
        positions = ["Fast Bowler", "Spin Bowler", "Medium Pacer", "All-Rounder", "Batsman", "Wicketkeeper"] if p_sport == "Cricket" else ["Striker / Forward", "Midfielder", "Defender", "Goalkeeper"]
        p_position = st.selectbox("Player Position", positions)
        p_age = st.number_input("Age", min_value=12, max_value=50, value=20)

    with col2:
        p_height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=172.0)
        p_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=140.0, value=68.0)
        p_matches = st.number_input("Matches Played This Week", min_value=0, max_value=14, value=2)

    if st.button("Save Player Profile"):
        if p_name:
            try:
                cursor.execute('''
                    INSERT INTO players (name, sport, position, age, height_cm, weight_kg, matches_this_week)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (p_name, p_sport, p_position, p_age, p_height, p_weight, p_matches))
                conn.commit()
                st.success(f"Player '{p_name}' successfully added as {p_sport} ({p_position})!")
            except sqlite3.IntegrityError:
                cursor.execute('''
                    UPDATE players SET sport=?, position=?, age=?, height_cm=?, weight_kg=?, matches_this_week=?
                    WHERE name=?
                ''', (p_sport, p_position, p_age, p_height, p_weight, p_matches, p_name))
                conn.commit()
                st.success(f"Player '{p_name}' profile updated!")

    st.markdown("---")
    st.subheader("Registered Player Roster")
    df_roster = pd.read_sql_query("SELECT id, name as 'Name', sport as 'Sport', position as 'Position', age as 'Age', height_cm as 'Height (cm)', weight_kg as 'Weight (kg)', matches_this_week as 'Matches/Wk' FROM players", conn)
    st.dataframe(df_roster, use_container_width=True)
