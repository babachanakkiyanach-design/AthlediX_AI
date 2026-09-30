import cv2
import time
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# Safe MediaPipe loading for Streamlit Cloud
import mediapipe as mp

try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except Exception:
    try:
        import mediapipe.python.solutions.pose as mp_pose
        import mediapipe.python.solutions.drawing_utils as mp_drawing
    except Exception as e:
        st.error(f"MediaPipe loading failed: {e}")

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
# 2. COLLEGE SQUAD DATABASES
# =========================================================
COLLEGE_DATABASE = {
    "Cricket": {
        "coach": "Raghav",
        "department": "Department of Physical Education",
        "players": {
            "Player 1": {
                "name": "Player 1 (Sathish)",
                "assigned_role": "Fast Bowler",
                "baseline_speed": 138.0,
                "yesterday_speed": 132.0,
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
                "baseline_speed": 28.5,
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
# 3. AI MODEL INITIALIZATION
# =========================================================
@st.cache_resource
def init_rf_model():
    data = [
        [1.1, 8.0, 2, 9, 0.0, 0],
        [1.2, 7.5, 3, 8, 2.0, 0],
        [1.4, 6.0, 6, 5, 12.0, 1],
        [1.5, 5.0, 8, 4, 18.0, 2],
        [1.6, 4.5, 9, 2, 25.0, 2],
        [0.9, 8.5, 1, 9, 1.0, 0],
        [1.35, 6.5, 5, 6, 8.0, 1]
    ]
    df = pd.DataFrame(data, columns=['ACWR', 'Sleep', 'Soreness', 'Readiness', 'Speed_Drop_Pct', 'Risk'])
    X = df[['ACWR', 'Sleep', 'Soreness', 'Readiness', 'Speed_Drop_Pct']]
    y = df['Risk']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

rf_model = init_rf_model()

# =========================================================
# 4. KINEMATICS & STANCE CLASSIFICATION
# =========================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def auto_classify_role_and_stance(landmarks, w, h):
    lwrist, rwrist = [landmarks[15].x * w, landmarks[15].y * h], [landmarks[16].x * w, landmarks[16].y * h]
    lshoulder, rshoulder = [landmarks[11].x * w, landmarks[11].y * h], [landmarks[12].x * w, landmarks[12].y * h]
    lknee = [landmarks[25].x * w, landmarks[25].y * h]
    lhip = [landmarks[23].x * w, landmarks[23].y * h]
    
    arm_span = abs(lwrist[0] - rwrist[0])
    shoulder_width = abs(lshoulder[0] - rshoulder[0])
    
    if arm_span > (shoulder_width * 2.0) and lwrist[1] < lhip[1]:
        return "GOALKEEPER (Ready Stance)"
    
    if lwrist[1] < lshoulder[1] - 20 or rwrist[1] < rshoulder[1] - 20:
        return "BOWLER (Delivery Stride)"
    
    knee_angle = calculate_angle(lhip, lknee, [landmarks[27].x * w, landmarks[27].y * h])
    if knee_angle < 150.0 and abs(lwrist[0] - rwrist[0]) < 80:
        return "BATTER (Batted Stance)"
        
    stride_gap = abs(landmarks[27].x - landmarks[28].x) * w
    if stride_gap > 70:
        return "RUNNER / SPRINTER (High Pace)"
        
    return "FIELDER / GENERAL ATHLETE"

def auto_detect_player(landmarks, image_height):
    hip, ankle = landmarks[23], landmarks[27]
    body_height_px = abs(ankle.y - hip.y) * image_height
    if body_height_px > 220:
        return "Player 1"
    elif body_height_px > 160:
        return "Player 2"
    else:
        return "Player 3"

def process_frame(frame, current_team_info):
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        player_key = auto_detect_player(landmarks, h)
        player_profile = current_team_info["players"][player_key]
        detected_stance_role = auto_classify_role_and_stance(landmarks, w, h)

        peak_speed = player_profile["baseline_speed"]
        yesterday_spd = player_profile["yesterday_speed"]
        speed_diff = peak_speed - yesterday_spd
        pct_change = (speed_diff / yesterday_spd) * 100.0
        
        if speed_diff >= 0:
            prog_str = f"IMPROVED (+{speed_diff:.1f} km/h, +{pct_change:.1f}% vs Yesterday)"
            prog_color = (0, 255, 0)
        else:
            prog_str = f"DROPPED ({speed_diff:.1f} km/h, {pct_change:.1f}% vs Yesterday)"
            prog_color = (0, 0, 255)

        speed_drop_pct = max(0.0, ((yesterday_spd - peak_speed) / yesterday_spd) * 100.0)
        input_vec = np.array([[player_profile['acwr'], player_profile['sleep'], player_profile['soreness'], player_profile['readiness'], speed_drop_pct]])
        risk_class = rf_model.predict(input_vec)[0]

        risk_map = {0: ("SAFE / OPTIMAL", (0, 255, 0)), 1: ("MODERATE FATIGUE", (0, 165, 255)), 2: ("HIGH INJURY RISK", (0, 0, 255))}
        risk_str, risk_color = risk_map[risk_class]

        # Draw HUD
        cv2.rectangle(frame, (10, 10), (w - 10, 140), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 140), risk_color, 2)

        cv2.putText(frame, f"COLLEGE: {INSTITUTION_NAME} | COACH: {current_team_info['coach']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"ATHLETE: {player_profile['name']}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"AUTO STANCE DETECTED: {detected_stance_role}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"Speed: {peak_speed} km/h | Yesterday: {yesterday_spd} km/h", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"ASSESSMENT: {prog_str}", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, prog_color, 2)

        return frame, detected_stance_role, prog_str
    return frame, "No Person Detected", "N/A"

# =========================================================
# 5. DASHBOARD LAYOUT & STREAMLIT UI
# =========================================================
st.title(f"🏫 {INSTITUTION_NAME}")
st.subheader("AthlediX AI — Auto-Stance, Role Detection & Assessment")

st.sidebar.header("🏆 Sport Selection")
selected_sport = st.sidebar.selectbox("Select Sport", ["Cricket", "Football"])
current_team_info = COLLEGE_DATABASE[selected_sport]

st.sidebar.markdown(f"**Head Coach:** {current_team_info['coach']}")
st.sidebar.markdown(f"**Department:** {current_team_info['department']}")
st.sidebar.divider()

st.sidebar.header("⚙️ Input Source")
input_mode = st.sidebar.radio(
    "Choose Input Method:", 
    ["📁 Upload Photo / Video (Recommended for Cloud)", "📸 Laptop / Mobile Webcam", "⚡ Run Synthetic Demo"]
)

tab_vision, tab_history = tab_vision, tab_history = st.tabs(["🎥 Live AI Processing", "📊 Yesterday vs. Today Assessment"])

with tab_vision:
    if input_mode == "📁 Upload Photo / Video (Recommended for Cloud)":
        st.info("Upload a video (.mp4, .mov) or image (.jpg, .png) of an athlete to perform AI pose analysis.")
        uploaded_file = st.file_uploader("Upload Media", type=["jpg", "jpeg", "png", "mp4", "mov"])

        if uploaded_file is not None:
            if uploaded_file.type.startswith("image"):
                image = Image.open(uploaded_file)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                processed_frame, stance, assessment = process_frame(frame, current_team_info)
                st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.success(f"**Stance Identified:** {stance} | **Status:** {assessment}")
            else:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame, stance, assessment = process_frame(frame, current_team_info)
                    st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                cap.release()

    elif input_mode == "📸 Laptop / Mobile Webcam":
        st.info("Click 'Take Photo' below to process a frame using your webcam.")
        img_file_buffer = st.camera_input("Capture Frame")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            processed_frame, stance, assessment = process_frame(cv_img, current_team_info)
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.success(f"**Auto-Detected Stance:** {stance} | **Assessment:** {assessment}")

    elif input_mode == "⚡ Run Synthetic Demo":
        st.success("Running synthetic AI field simulation...")
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Simulate athlete skeleton
        cv2.circle(blank_frame, (320, 120), 20, (255, 255, 255), -1)
        cv2.line(blank_frame, (320, 140), (320, 280), (255, 255, 255), 4)
        cv2.line(blank_frame, (320, 180), (250, 220), (255, 255, 255), 4)
        cv2.line(blank_frame, (320, 180), (390, 220), (255, 255, 255), 4)
        cv2.line(blank_frame, (320, 280), (280, 380), (255, 255, 255), 4)
        cv2.line(blank_frame, (320, 280), (360, 380), (255, 255, 255), 4)

        # Draw HUD
        cv2.rectangle(blank_frame, (10, 10), (630, 140), (20, 20, 20), -1)
        cv2.rectangle(blank_frame, (10, 10), (630, 140), (0, 255, 0), 2)
        cv2.putText(blank_frame, f"COLLEGE: {INSTITUTION_NAME} | COACH: {current_team_info['coach']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(blank_frame, f"ATHLETE: Player 1 (Sathish)", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(blank_frame, f"AUTO STANCE DETECTED: BOWLER (Delivery Stride)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(blank_frame, f"Speed: 138.0 km/h | Yesterday: 132.0 km/h", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(blank_frame, f"ASSESSMENT: IMPROVED (+6.0 km/h, +4.5% vs Yesterday)", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        st.image(cv2.cvtColor(blank_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

with tab_history:
    st.markdown(f"### 📊 Yesterday vs. Today Assessment — {selected_sport} Squad")
    players = current_team_info["players"]
    cols = st.columns(len(players))

    for idx, (p_key, p_data) in enumerate(players.items()):
        with cols[idx]:
            st.markdown(f"#### {p_data['name']}")
            st.caption(f"Role: `{p_data['assigned_role']}`")
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

    st.divider()
    rows = []
    for p_key, p in players.items():
        diff = round(p["baseline_speed"] - p["yesterday_speed"], 1)
        status = "🟢 Improved" if diff >= 0 else "🔴 Performance Drop"
        rows.append({
            "Athlete": p["name"],
            "Position": p["assigned_role"],
            "Today Speed": f"{p['baseline_speed']} km/h",
            "Yesterday Speed": f"{p['yesterday_speed']} km/h",
            "Progress": f"{diff:+} km/h",
            "Assessment": status,
            "ACWR": p["acwr"]
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
