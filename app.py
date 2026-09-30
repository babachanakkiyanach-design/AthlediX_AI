import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import os
import sqlite3
from collections import deque, defaultdict
from datetime import datetime
from ultralytics import YOLO

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
    technique_metric REAL,
    ai_fatigue INTEGER,
    readiness_score REAL,
    performance_score REAL
)
''')
conn.commit()

# =========================================================
# 2. LOAD YOLOV8 MODEL
# =========================================================
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n-pose.pt")

model = load_yolo_model()

# =========================================================
# 3. HELPER FUNCTIONS & BIOMECHANICS ENGINE
# =========================================================
def calculate_angle(a, b, c):
    """Calculates joint angle in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return int(angle)

def calculate_ai_fatigue(angle_variance, speed_decay):
    """
    AI Automatic Fatigue Scoring (1 = Fresh, 5 = Exhausted)
    Based on joint stability variance and velocity drop-off.
    """
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
    """Calculates Match Readiness Score using AI fatigue index (0% - 100%)."""
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
st.set_page_config(page_title="AthlediX AI Multi-Bowler Engine", layout="wide", page_icon="🏆")

st.title("🏆 AthlediX AI: Multi-Bowler Tracking & Automatic AI Biomechanics")
st.markdown("Multi-Bowler Real-Speed Tracking, Automatic AI Fatigue Detection & ICC Arm Legality Rules.")

tab1, tab2, tab3 = st.tabs(["📹 Multi-Bowler Motion Analysis", "📅 Day-by-Day History", "👤 Registered Roster"])

cursor.execute("SELECT name FROM players")
registered_players = [row[0] for row in cursor.fetchall()]

# Bounding box color palette for multiple bowlers
BOWLER_COLORS = [
    (0, 255, 0),    # Lime Green
    (255, 165, 0),  # Orange
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (0, 128, 255)   # Blue-Orange
]

# ---------------------------------------------------------
# TAB 1: MULTI-BOWLER VIDEO ANALYSIS
# ---------------------------------------------------------
with tab1:
    st.header("Upload Video (Single Bowler or Multi-Bowler Clip)")
    
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_video = st.file_uploader("Upload Bowling Clip (MP4 / MOV / AVI)", type=["mp4", "mov", "avi"])
    with col_u2:
        if registered_players:
            primary_player = st.selectbox("Primary Tagged Player (Optional)", registered_players)
        else:
            primary_player = st.text_input("Primary Athlete Name", value="Guest Bowler")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        input_video_path = tfile.name

        st.subheader("Raw Uploaded Clip")
        st.video(input_video_path)

        if st.button("🚀 Analyze All Bowlers & Find the Fastest"):
            with st.spinner("Tracking multiple athletes and processing AI fatigue mechanics..."):
                cap = cv2.VideoCapture(input_video_path)
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0 or np.isnan(fps):
                    fps = 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                # Multi-Bowler tracking structures
                bowler_speeds = defaultdict(list)
                bowler_angles = defaultdict(list)
                prev_wrists = {}
                prev_times = {}
                wrist_trails = defaultdict(lambda: deque(maxlen=15))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    # Run YOLO Multi-Object Pose Tracking
                    results = model.track(frame, persist=True, verbose=False)

                    if results and len(results[0].keypoints) > 0 and results[0].boxes.id is not None:
                        boxes = results[0].boxes
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        keypoints_data = results[0].keypoints.data.cpu().numpy()

                        for idx, track_id in enumerate(track_ids):
                            kpts = keypoints_data[idx]

                            if len(kpts) >= 11:
                                # Determine dynamic scale: height of torso/body in pixels to calculate real meters
                                nose, ankle = kpts[0][:2], kpts[16][:2] if len(kpts) > 16 else kpts[10][:2]
                                person_pixel_height = np.abs(ankle[1] - nose[1])
                                if person_pixel_height > 50:
                                    meters_per_px = 1.70 / float(person_pixel_height)
                                else:
                                    meters_per_px = 1.70 / 380.0

                                # Check Right vs Left arm visibility
                                r_conf = kpts[6][2] + kpts[8][2] + kpts[10][2]
                                l_conf = kpts[5][2] + kpts[7][2] + kpts[9][2]

                                if r_conf >= l_conf and kpts[10][2] > 0.30:
                                    s_pt, e_pt, w_pt = kpts[6][:2], kpts[8][:2], kpts[10][:2]
                                elif l_conf > r_conf and kpts[9][2] > 0.30:
                                    s_pt, e_pt, w_pt = kpts[5][:2], kpts[7][:2], kpts[9][:2]
                                else:
                                    s_pt, e_pt, w_pt = None, None, None

                                if s_pt is not None and e_pt is not None and w_pt is not None:
                                    sx, sy = int(s_pt[0]), int(s_pt[1])
                                    ex, ey = int(e_pt[0]), int(e_pt[1])
                                    wx, wy = int(w_pt[0]), int(w_pt[1])

                                    angle = calculate_angle(s_pt, e_pt, w_pt)
                                    bowler_angles[track_id].append(angle)

                                    curr_speed = 0.0
                                    if track_id in prev_wrists and track_id in prev_times:
                                        dt = curr_time - prev_times[track_id]
                                        if dt > 0.005:
                                            dist_px = np.sqrt((w_pt[0] - prev_wrists[track_id][0])**2 + (w_pt[1] - prev_wrists[track_id][1])**2)
                                            speed_kmh = (dist_px * meters_per_px / dt) * 3.6

                                            # Bounded real bowling speeds (15 km/h to 165 km/h)
                                            if 15.0 < speed_kmh < 165.0:
                                                bowler_speeds[track_id].append(speed_kmh)
                                                curr_speed = speed_kmh

                                    prev_wrists[track_id] = w_pt
                                    prev_times[track_id] = curr_time
                                    wrist_trails[track_id].append((wx, wy))

                                    # Pick visual color for bowler ID
                                    color = BOWLER_COLORS[(track_id - 1) % len(BOWLER_COLORS)]

                                    # Draw Skeleton Lines
                                    cv2.line(frame, (sx, sy), (ex, ey), color, 3)
                                    cv2.line(frame, (ex, ey), (wx, wy), color, 3)
                                    cv2.circle(frame, (wx, wy), 8, (0, 0, 255), -1)

                                    # Motion trail
                                    for t_idx in range(1, len(wrist_trails[track_id])):
                                        cv2.line(frame, wrist_trails[track_id][t_idx - 1], wrist_trails[track_id][t_idx], color, 2)

                                    # Label Box Above Head
                                    label_text = f"Bowler #{track_id}: {curr_speed:.1f} km/h | {angle} deg"
                                    cv2.putText(frame, label_text, (sx - 20, max(20, sy - 15)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    # Top Overlay Banner
                    cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
                    cv2.putText(frame, "AthlediX AI: Multi-Bowler Tracking Active",
                                (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    out.write(frame)

                cap.release()
                out.release()

                st.markdown("---")
                st.subheader("🎬 AI Annotated Multi-Bowler Output")
                st.video(output_video_path)

                # Process multi-bowler rankings
                summary_data = []
                fastest_id = None
                highest_speed = -1.0

                for b_id, speeds in bowler_speeds.items():
                    peak_s = max(speeds) if speeds else 0.0
                    angles = bowler_angles[b_id]
                    avg_a = np.mean(angles) if angles else 0.0
                    ang_var = float(np.std(angles)) if angles else 0.0
                    
                    # Calculate AI Automatic Fatigue
                    spd_decay = (speeds[0] - speeds[-1]) if len(speeds) > 3 else 0.0
                    ai_fatigue = calculate_ai_fatigue(ang_var, max(0.0, spd_decay))

                    # Check ICC 15-degree Rule (Max extension flex)
                    elbow_flex = (max(angles) - min(angles)) if angles else 0.0
                    icc_status = "⚠️ Non-Compliant (>15° flex)" if elbow_flex > 15.0 else "✅ Legal Action"

                    if peak_s > highest_speed:
                        highest_speed = peak_s
                        fastest_id = b_id

                    summary_data.append({
                        "Bowler ID": f"Bowler #{b_id}",
                        "Peak Speed (km/h)": round(peak_s, 1),
                        "Avg Arm Angle (deg)": round(avg_a, 1),
                        "Elbow Flex Change": f"{elbow_flex:.1f}°",
                        "ICC Arm Legality": icc_status,
                        "AI Fatigue Score (1-5)": f"{ai_fatigue} / 5"
                    })

                # Display Multi-Bowler Leaderboard Table
                st.subheader("⚡ Multi-Bowler Speed & AI Fatigue Breakdown")
                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True)

                    st.success(f"🔥 **Fastest Bowler in Clip:** **Bowler #{fastest_id}** with a peak speed of **{highest_speed:.1f} km/h**!")

                    # Save Primary Tagged Player performance to database
                    cursor.execute("SELECT sport, position, height_cm, weight_kg, matches_this_week FROM players WHERE name = ?", (primary_player,))
                    p_prof = cursor.fetchone()

                    if p_prof:
                        sport, position, height, weight, matches = p_prof
                    else:
                        sport, position, height, weight, matches = "Cricket", "Fast Bowler", 175.0, 70.0

                    primary_speed = highest_speed
                    primary_angle = summary_data[0]["Avg Arm Angle (deg)"] if summary_data else 165.0
                    primary_fatigue = int(summary_data[0]["AI Fatigue Score (1-5)"].split()[0]) if summary_data else 2
                    
                    readiness = calculate_readiness_score(height, weight, matches, primary_fatigue)
                    perf_score = round(min(60.0, (primary_speed / 150.0) * 60.0) + (40.0 if primary_angle > 150 else 30.0), 1)

                    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute('''
                        INSERT INTO performance_history 
                        (player_name, sport, position, timestamp, peak_speed, technique_metric, ai_fatigue, readiness_score, performance_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (primary_player, sport, position, current_time_str, primary_speed, primary_angle, primary_fatigue, readiness, perf_score))
                    conn.commit()

# ---------------------------------------------------------
# TAB 2: DAY-BY-DAY HISTORY
# ---------------------------------------------------------
with tab2:
    st.header("📅 Day-by-Day Performance History")
    df_filtered = pd.read_sql_query("SELECT timestamp as 'Date & Time', player_name as 'Athlete', sport as 'Sport', position as 'Position', peak_speed as 'Peak Speed (km/h)', ai_fatigue as 'AI Fatigue (1-5)', readiness_score as 'Match Readiness (%)', performance_score as 'Performance Score (100)' FROM performance_history ORDER BY timestamp DESC", conn)
    
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

    st.info("ℹ️ **Fatigue Level Notice:** Manual fatigue input has been disabled. The AI Engine automatically detects fatigue during video processing.")

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
