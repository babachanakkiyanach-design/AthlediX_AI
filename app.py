import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 1. PAGE CONFIGURATION & INSTITUTION BRANDING
# =========================================================
st.set_page_config(
    page_title="AVS Engineering College - AthlediX AI",
    page_icon="⚡",
    layout="wide"
)

INSTITUTION_NAME = "AVS Engineering College"

# =========================================================
# 2. COLLEGE SQUAD DATABASES (CRICKET & FOOTBALL + HISTORICAL)
# =========================================================
COLLEGE_DATABASE = {
    "Cricket": {
        "coach": "Raghav",
        "department": "Department of Physical Education",
        "players": {
            "Player 1": {
                "name": "Player 1 (Sathish)",
                "assigned_role": "Fast Bowler",
                "baseline_speed": 138.0,      # km/h
                "yesterday_speed": 132.0,     # Yesterday's performance
                "acwr": 1.15, "sleep": 8.0, "soreness": 2, "readiness": 9,
                "matches": 28, "stats": "42 Wickets"
            },
            "Player 2": {
                "name": "Player 2 (Kumar)",
                "assigned_role": "Batter / All-Rounder",
                "baseline_speed": 125.0,
                "yesterday_speed": 128.0,
                "acwr": 1.48, "sleep": 5.5, "soreness": 7, "readiness": 5,
                "matches": 35, "stats": "620 Runs"
            },
            "Player 3": {
                "name": "Player 3 (Sowmiya)",
                "assigned_role": "Spin Bowler",
                "baseline_speed": 108.0,
                "yesterday_speed": 104.0,
                "acwr": 1.05, "sleep": 7.5, "soreness": 3, "readiness": 8,
                "matches": 22, "stats": "31 Wickets"
            }
        }
    },
    "Football": {
        "coach": "Karthik",
        "department": "Department of Physical Education",
        "players": {
            "Player 1": {
                "name": "Player 1 (Nisha)",
                "assigned_role": "Forward Player",
                "baseline_speed": 28.5,       # km/h sprint
                "yesterday_speed": 26.0,
                "acwr": 1.22, "sleep": 7.5, "soreness": 3, "readiness": 8,
                "matches": 31, "stats": "18 Goals"
            },
            "Player 2": {
                "name": "Player 2 (Thillai)",
                "assigned_role": "Goalkeeper",
                "baseline_speed": 22.0,
                "yesterday_speed": 23.5,
                "acwr": 1.52, "sleep": 5.0, "soreness": 8, "readiness": 4,
                "matches": 40, "stats": "14 Clean Sheets"
            },
            "Player 3": {
                "name": "Player 3 (Ramesh)",
                "assigned_role": "Defender / Runner",
                "baseline_speed": 25.2,
                "yesterday_speed": 24.0,
                "acwr": 0.98, "sleep": 8.5, "soreness": 2, "readiness": 9,
                "matches": 36, "stats": "5 Assists"
            }
        }
    }
}

# =========================================================
# 3. RANDOM FOREST AI MODEL INITIALIZATION
# =========================================================
@st.cache_resource
def init_rf_model():
    data = [
        [1.1, 8.0, 2, 9, 0.0, 0],   # Safe
        [1.2, 7.5, 3, 8, 2.0, 0],   # Safe
        [1.4, 6.0, 6, 5, 12.0, 1],  # Warning
        [1.5, 5.0, 8, 4, 18.0, 2],  # High Risk
        [1.6, 4.5, 9, 2, 25.0, 2],  # High Risk
        [0.9, 8.5, 1, 9, 1.0, 0],   # Safe
        [1.35, 6.5, 5, 6, 8.0, 1]   # Warning
    ]
    df = pd.DataFrame(data, columns=['ACWR', 'Sleep', 'Soreness', 'Readiness', 'Speed_Drop_Pct', 'Risk'])
    X = df[['ACWR', 'Sleep', 'Soreness', 'Readiness', 'Speed_Drop_Pct']]
    y = df['Risk']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

rf_model = init_rf_model()

# =========================================================
# 4. KINEMATICS, STANCE CLASSIFICATION & DELTA MATH
# =========================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def auto_classify_role_and_stance(landmarks, w, h):
    """Detects if player is standing/acting like a Goalkeeper, Bowler, Batter, or Runner."""
    lwrist, rwrist = [landmarks[15].x * w, landmarks[15].y * h], [landmarks[16].x * w, landmarks[16].y * h]
    lshoulder, rshoulder = [landmarks[11].x * w, landmarks[11].y * h], [landmarks[12].x * w, landmarks[12].y * h]
    lknee, rknee = [landmarks[25].x * w, landmarks[25].y * h], [landmarks[26].x * w, landmarks[26].y * h]
    lhip, rhip = [landmarks[23].x * w, landmarks[23].y * h], [landmarks[24].x * w, landmarks[24].y * h]
    
    # 1. GOALKEEPER STANCE (Arms spread wide horizontally & low knees)
    arm_span = abs(lwrist[0] - rwrist[0])
    shoulder_width = abs(lshoulder[0] - rshoulder[0])
    if arm_span > (shoulder_width * 2.2) and lwrist[1] < lhip[1]:
        return "GOALKEEPER (Ready Stance)"
    
    # 2. BOWLER STANCE / ACTION (One arm extended high above shoulder level)
    if lwrist[1] < lshoulder[1] - 30 or rwrist[1] < rshoulder[1] - 30:
        return "BOWLER (Delivery Stride)"
    
    # 3. BATTER / CROUCH STANCE (Knees bent, hands low together)
    knee_angle = calculate_angle(lhip, lknee, [landmarks[27].x * w, landmarks[27].y * h])
    if knee_angle < 145.0 and abs(lwrist[0] - rwrist[0]) < 60:
        return "BATTER (Batted Stance)"
        
    # 4. RUNNER / SPRINTER (Wide stride separation between knees/feet)
    stride_gap = abs(landmarks[27].x - landmarks[28].x) * w
    if stride_gap > 80:
        return "RUNNER / SPRINTER (High Pace)"
        
    return "FIELDER / GENERAL ATHLETE"

def auto_detect_player(landmarks, image_height):
    hip, ankle = landmarks[23], landmarks[27]
    body_height_px = abs(ankle.y - hip.y) * image_height
    if body_height_px > 230:
        return "Player 1"
    elif body_height_px > 170:
        return "Player 2"
    else:
        return "Player 3"

# =========================================================
# 5. DASHBOARD LAYOUT & STREAMLIT UI
# =========================================================
st.title(f"🏫 {INSTITUTION_NAME}")
st.subheader("AthlediX AI — Auto-Stance, Role Detection & Yesterday vs. Today Comparison")

st.sidebar.header("🏆 Sport Selection")
selected_sport = st.sidebar.selectbox("Select Sport", ["Cricket", "Football"])
current_team_info = COLLEGE_DATABASE[selected_sport]

st.sidebar.markdown(f"**Head Coach:** {current_team_info['coach']}")
st.sidebar.markdown(f"**Department:** {current_team_info['department']}")
st.sidebar.divider()

st.sidebar.header("📷 Camera Setup")
camera_source = st.sidebar.radio("Camera Source", ["Laptop Webcam", "Mobile Hotspot IP Camera"])

if camera_source == "Laptop Webcam":
    video_url = 0
else:
    ip_address = st.sidebar.text_input("Mobile IP Camera URL", value="http://192.168.43.1:8080/video")
    video_url = ip_address

tab_vision, tab_history = st.tabs(["🎥 Live AI Vision & Assessment", "📊 Yesterday vs. Today Assessment"])

# =========================================================
# TAB 1: LIVE COMPUTER VISION ENGINE WITH YESTERDAY DELTA
# =========================================================
with tab_vision:
    st.markdown("### Live Field AI Feed")
    run_feed = st.checkbox("Turn On Live Camera Feed", value=True)
    frame_placeholder = st.empty()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    if run_feed:
        cap = cv2.VideoCapture(video_url)
        prev_time = time.time()
        prev_wrist_pos = None
        peak_speed = 0.0
        last_action_time = time.time()

        while cap.isOpened() and run_feed:
            ret, frame = cap.read()
            if not ret:
                st.warning("Connecting to camera stream... Check Hotspot or IP URL.")
                time.sleep(0.5)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 1. AUTO PLAYER DETECTION & ROLE ASSIGNMENT
                player_key = auto_detect_player(landmarks, h)
                player_profile = current_team_info["players"][player_key]

                # 2. AUTO STANCE / ROLE DETECTED BY CAMERA
                detected_stance_role = auto_classify_role_and_stance(landmarks, w, h)

                # 3. SPEED & MOTION METRICS
                shoulder = [landmarks[12].x * w, landmarks[12].y * h]
                elbow = [landmarks[14].x * w, landmarks[14].y * h]
                wrist = [landmarks[16].x * w, landmarks[16].y * h]

                curr_wrist_pos = np.array(wrist)
                if prev_wrist_pos is not None and dt > 0:
                    dist_px = np.linalg.norm(curr_wrist_pos - prev_wrist_pos)
                    vel_px_sec = dist_px / dt
                    measured = vel_px_sec * 0.16
                    if measured > 30.0:
                        peak_speed = round(measured, 1)
                        last_action_time = curr_time

                prev_wrist_pos = curr_wrist_pos

                if curr_time - last_action_time > 4.0:
                    peak_speed = player_profile["baseline_speed"]

                # 4. YESTERDAY VS TODAY PROGRESS COMPARISON
                yesterday_spd = player_profile["yesterday_speed"]
                speed_diff = peak_speed - yesterday_spd
                pct_change = (speed_diff / yesterday_spd) * 100.0
                
                if speed_diff >= 0:
                    prog_str = f"IMPROVED (+{speed_diff:.1f} km/h, +{pct_change:.1f}% vs Yesterday)"
                    prog_color = (0, 255, 0)
                else:
                    prog_str = f"DROPPED ({speed_diff:.1f} km/h, {pct_change:.1f}% vs Yesterday)"
                    prog_color = (0, 0, 255)

                # 5. RANDOM FOREST RISK PIPELINE
                base_speed = player_profile["baseline_speed"]
                speed_drop_pct = max(0.0, ((base_speed - peak_speed) / base_speed) * 100.0)

                input_vec = np.array([[player_profile['acwr'], player_profile['sleep'], player_profile['soreness'], player_profile['readiness'], speed_drop_pct]])
                risk_class = rf_model.predict(input_vec)[0]

                risk_map = {0: ("SAFE / OPTIMAL", (0, 255, 0)), 1: ("MODERATE FATIGUE", (0, 165, 255)), 2: ("HIGH INJURY RISK", (0, 0, 255))}
                risk_str, risk_color = risk_map[risk_class]

                # 6. DRAW ON-SCREEN HUD
                cv2.rectangle(frame, (10, 10), (w - 10, 150), (20, 20, 20), -1)
                cv2.rectangle(frame, (10, 10), (w - 10, 150), risk_color, 2)

                cv2.putText(frame, f"COLLEGE: {INSTITUTION_NAME} | COACH: {current_team_info['coach']}", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(frame, f"ATHLETE: {player_profile['name']}", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"AUTO STANCE DETECTED: {detected_stance_role}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(frame, f"Live Speed: {peak_speed} km/h | Yesterday: {yesterday_spd} km/h", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"ASSESSMENT: {prog_str}", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, prog_color, 2)

                # Risk Badge
                cv2.rectangle(frame, (w - 260, 20), (w - 20, 70), risk_color, -1)
                cv2.putText(frame, risk_str, (w - 250, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()

# =========================================================
# TAB 2: YESTERDAY VS TODAY COMPARATIVE DATABASE
# =========================================================
with tab_history:
    st.markdown(f"### 📊 Yesterday vs. Today Assessment — {selected_sport} Squad")
    st.caption("Automated tracking comparing live camera measurements against historical baselines.")

    players = current_team_info["players"]
    cols = st.columns(len(players))

    for idx, (p_key, p_data) in enumerate(players.items()):
        with cols[idx]:
            st.markdown(f"#### {p_data['name']}")
            st.caption(f"Role: `{p_data['assigned_role']}`")
            
            # Metric Progress Delta vs Yesterday
            spd_today = p_data["baseline_speed"]
            spd_yest = p_data["yesterday_speed"]
            delta_val = round(spd_today - spd_yest, 1)
            
            st.metric(
                label="Today Speed vs Yesterday", 
                value=f"{spd_today} km/h", 
                delta=f"{delta_val} km/h vs Yesterday"
            )
            st.metric(label="Matches Played", value=p_data["matches"])
            st.metric(label="Primary Performance", value=p_data["stats"])
            st.markdown(f"**ACWR Workload:** `{p_data['acwr']}`")
            st.markdown(f"**Sleep:** `{p_data['sleep']} hrs` | **Soreness:** `{p_data['soreness']}/10`")

    st.divider()
    
    # Master Table
    rows = []
    for p_key, p in players.items():
        diff = round(p["baseline_speed"] - p["yesterday_speed"], 1)
        status = "🟢 Improved" if diff >= 0 else "🔴 Performance Drop"
        rows.append({
            "Athlete": p["name"],
            "Assigned Position": p["assigned_role"],
            "Today Speed": f"{p['baseline_speed']} km/h",
            "Yesterday Speed": f"{p['yesterday_speed']} km/h",
            "Speed Progress": f"{diff:+} km/h",
            "Assessment": status,
            "ACWR": p["acwr"],
            "Soreness": f"{p['soreness']}/10"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
