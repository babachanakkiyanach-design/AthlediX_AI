import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import os
import sqlite3
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
    matches_this_week INTEGER DEFAULT 0,
    fatigue_level INTEGER DEFAULT 1,
    photo_path TEXT
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
# 3. HELPER FUNCTIONS
# =========================================================
def calculate_angle(a, b, c):
    """Calculates joint angle in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return int(angle)

def calculate_readiness_score(height_cm, weight_kg, matches_this_week, fatigue_level):
    """Calculates Match Readiness Score (0% - 100%)."""
    base_readiness = 100.0
    workload_deduction = matches_this_week * 7.5
    fatigue_deduction = (fatigue_level - 1) * 10.0
    
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2) if height_m > 0 else 22.0
    bmi_penalty = 5.0 if (bmi < 18.5 or bmi > 25.0) else 0.0

    readiness = base_readiness - workload_deduction - fatigue_deduction - bmi_penalty
    return max(0.0, min(100.0, round(readiness, 1)))

# =========================================================
# 4. STREAMLIT APP UI
# =========================================================
st.set_page_config(page_title="AthlediX AI Engine", layout="wide", page_icon="🏆")

st.title("🏆 AthlediX AI: Performance & Readiness Engine")
st.markdown("YOLOv8 Pose Motion Tracking, Bowling Speed Analysis & Player Readiness Leaderboards.")

tab1, tab2, tab3 = st.tabs(["📹 Video Analysis & Best Performer", "📅 Day-by-Day History", "👤 Player & Readiness Roster"])

# Fetch player names for selection dropdowns
cursor.execute("SELECT name FROM players")
registered_players = [row[0] for row in cursor.fetchall()]

# ---------------------------------------------------------
# TAB 1: VIDEO ANALYSIS
# ---------------------------------------------------------
with tab1:
    st.header("Upload Player Performance Video")
    
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_video = st.file_uploader("Upload Bowling / Athletic Video (MP4 / MOV)", type=["mp4", "mov", "avi"])
    with col_u2:
        if registered_players:
            selected_player = st.selectbox("Select Athlete in Video", registered_players)
        else:
            selected_player = st.text_input("Enter Athlete Name", value="Guest Player")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("🚀 Analyze Motion & Record Performance"):
            with st.spinner("Analyzing motion keypoints with Ultralytics YOLOv8..."):
                cap = cv2.VideoCapture(video_path)
                
                speeds = []
                angles = []
                prev_wrist = None
                prev_time = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                    results = model(frame, verbose=False)
                    if results and len(results[0].keypoints) > 0:
                        kpts = results[0].keypoints.data.cpu().numpy()[0]

                        if len(kpts) >= 11:
                            r_shoulder, r_elbow, r_wrist = kpts[6][:2], kpts[8][:2], kpts[10][:2]

                            if kpts[6][2] > 0.35 and kpts[8][2] > 0.35 and kpts[10][2] > 0.35:
                                angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                                angles.append(angle)

                            if kpts[10][2] > 0.35:
                                if prev_wrist is not None and prev_time is not None:
                                    dt = curr_time - prev_time
                                    if dt > 0.005:
                                        dist_px = np.sqrt((r_wrist[0] - prev_wrist[0])**2 + (r_wrist[1] - prev_wrist[1])**2)
                                        meters_per_px = 1.7 / 380.0
                                        speed_kmh = (dist_px * meters_per_px / dt) * 3.6

                                        if 12.0 < speed_kmh < 160.0:
                                            speeds.append(speed_kmh)

                                prev_wrist = r_wrist
                                prev_time = curr_time

                cap.release()

                # Get Player Profile Info
                cursor.execute("SELECT sport, position, height_cm, weight_kg, matches_this_week, fatigue_level FROM players WHERE name = ?", (selected_player,))
                player_profile = cursor.fetchone()

                if player_profile:
                    sport, position, height, weight, matches, fatigue = player_profile
                    readiness = calculate_readiness_score(height, weight, matches, fatigue)
                else:
                    sport, position, readiness = "General", "Athlete", 85.0

                top_speed = round(max(speeds), 1) if speeds else 0.0
                avg_angle = round(np.mean(angles), 1) if angles else 0.0
                
                speed_score = min(60.0, (top_speed / 140.0) * 60.0)
                tech_score = 40.0 if avg_angle > 150 else (avg_angle / 150.0) * 40.0
                total_perf_score = round(speed_score + tech_score, 1)

                current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute('''
                    INSERT INTO performance_history 
                    (player_name, sport, position, timestamp, peak_speed, technique_metric, readiness_score, performance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (selected_player, sport, position, current_time_str, top_speed, avg_angle, readiness, total_perf_score))
                conn.commit()

                st.subheader("🎯 Session Results")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("👤 Athlete", selected_player)
                m2.metric("⚽ Sport / Role", f"{sport} ({position})")
                m3.metric("⚡ Peak Speed", f"{top_speed} km/h")
                m4.metric("💚 Match Readiness", f"{readiness}%")
                m5.metric("⭐ Performance Rating", f"{total_perf_score} / 100")

                st.success(f"Performance logged on {current_time_str}!")

    st.markdown("---")
    st.header("🌟 Best Performer Evaluation")
    df_history = pd.read_sql_query("SELECT * FROM performance_history ORDER BY performance_score DESC", conn)
    
    if not df_history.empty:
        best = df_history.iloc[0]
        st.info(f"🏆 **Top Best Performer:** **{best['player_name']}** ({best['sport']} - {best['position']}) with a Performance Score of **{best['performance_score']}/100**, Speed of **{best['peak_speed']} km/h**, and Match Readiness of **{best['readiness_score']}%**!")

# ---------------------------------------------------------
# TAB 2: DAY-BY-DAY HISTORY
# ---------------------------------------------------------
with tab2:
    st.header("📅 Day-by-Day Performance History")
    df_filtered = pd.read_sql_query("SELECT timestamp as 'Date & Time', player_name as 'Athlete', sport as 'Sport', position as 'Position', peak_speed as 'Peak Speed (km/h)', readiness_score as 'Match Readiness (%)', performance_score as 'Performance Score (100)' FROM performance_history ORDER BY timestamp DESC", conn)
    st.dataframe(df_filtered, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: REGISTER PLAYER
# ---------------------------------------------------------
with tab3:
    st.header("Add Player Profile & Calculate Readiness")
    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Player Name (e.g., Sweety / Priya)")
        p_sport = st.selectbox("Sport Category", ["Cricket", "Football"])
        positions = ["Fast Bowler", "Spin Bowler", "All-Rounder", "Batsman", "Wicketkeeper"] if p_sport == "Cricket" else ["Striker / Forward", "Midfielder", "Defender", "Goalkeeper"]
        p_position = st.selectbox("Player Position", positions)
        p_age = st.number_input("Age", min_value=12, max_value=50, value=20)
        p_height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=172.0)

    with col2:
        p_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=140.0, value=68.0)
        p_matches = st.number_input("Matches Played This Week", min_value=0, max_value=14, value=2)
        p_fatigue = st.slider("Fatigue Level (1 = Fresh, 5 = Exhausted)", 1, 5, 2)

    calculated_readiness = calculate_readiness_score(p_height, p_weight, p_matches, p_fatigue)
    st.info(f"💡 Calculated Initial Match Readiness: **{calculated_readiness}%**")

    if st.button("Save Player Profile"):
        if p_name:
            try:
                cursor.execute('''
                    INSERT INTO players (name, sport, position, age, height_cm, weight_kg, matches_this_week, fatigue_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (p_name, p_sport, p_position, p_age, p_height, p_weight, p_matches, p_fatigue))
                conn.commit()
                st.success(f"Player '{p_name}' successfully added!")
            except sqlite3.IntegrityError:
                cursor.execute('''
                    UPDATE players SET sport=?, position=?, age=?, height_cm=?, weight_kg=?, matches_this_week=?, fatigue_level=?
                    WHERE name=?
                ''', (p_sport, p_position, p_age, p_height, p_weight, p_matches, p_fatigue, p_name))
                conn.commit()
                st.success(f"Player '{p_name}' profile updated!")

    st.markdown("---")
    st.subheader("Registered Player Roster")
    df_roster = pd.read_sql_query("SELECT id, name as 'Name', sport as 'Sport', position as 'Position', age as 'Age', height_cm as 'Height (cm)', weight_kg as 'Weight (kg)', matches_this_week as 'Matches/Wk', fatigue_level as 'Fatigue' FROM players", conn)
    st.dataframe(df_roster, use_container_width=True)
