import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp

# =========================================================
# 1. CLOUD-SAFE MEDIAPIPE INITIALIZATION
# =========================================================
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Direct import fallback for Linux/Streamlit Cloud runtime
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="AVS College Sports AI", layout="wide", page_icon="🏆")

# =========================================================
# 3. FULL 11-PLAYER SQUAD DATABASES (TAMIL NADU CONTEXT)
# =========================================================
def generate_squad(sport, gender):
    if sport == "Cricket" and gender == "Boys":
        names = ["Ashwin", "Karthik", "Vijay", "Surya", "Dinesh", "Washington", "Natarajan", "Sandeep", "Sai", "Shahrukh", "Varun"]
        positions = ["Opening Batter", "Opening Batter", "Top Order", "Middle Order", "Wicketkeeper", "All-Rounder", "Fast Bowler", "Fast Bowler", "Spin Bowler", "Spin Bowler", "Fast Bowler"]
        speeds = [110, 105, 115, 120, 100, 125, 142, 138, 95, 92, 135]
    elif sport == "Cricket" and gender == "Girls":
        names = ["Hemalatha", "Niranjana", "Anusha", "Keerthana", "Ramyashri", "Nethra", "Shailaja", "Akshaya", "Arshi", "Kavya", "Priya"]
        positions = ["Opening Batter", "Opening Batter", "Top Order", "Middle Order", "Wicketkeeper", "All-Rounder", "Fast Bowler", "Fast Bowler", "Spin Bowler", "Spin Bowler", "Fast Bowler"]
        speeds = [95, 92, 98, 100, 85, 105, 120, 118, 80, 82, 115]
    elif sport == "Football" and gender == "Boys":
        names = ["Nandha", "Edwin", "Sivasakthi", "Prasanth", "Michael", "Soosairaj", "Jesuraj", "Charles", "Dhanpal", "Raegan", "Kamal"]
        positions = ["Goalkeeper", "Center Back", "Center Back", "Left Back", "Right Back", "Defensive Mid", "Central Mid", "Attacking Mid", "Left Winger", "Right Winger", "Striker"]
        speeds = [22, 26, 27, 29, 28, 25, 27, 28, 32, 31, 33]
    else:  # Football Girls
        names = ["Indumathi", "Sandhiya", "Karthika", "Sumithra", "Kausalya", "Pavithra", "Durga", "Mariyammal", "Sowmiya", "Vinitha", "Nandhini"]
        positions = ["Goalkeeper", "Center Back", "Center Back", "Left Back", "Right Back", "Defensive Mid", "Central Mid", "Attacking Mid", "Left Winger", "Right Winger", "Striker"]
        speeds = [18, 22, 23, 25, 24, 21, 23, 24, 28, 27, 29]

    return pd.DataFrame({
        "Jersey": list(range(1, 12)),
        "Player Name": names,
        "Position": positions,
        "Baseline Speed (km/h)": speeds,
        "Form / Readiness": [9, 8, 9, 10, 8, 9, 10, 9, 8, 9, 8],
        "Matches Played": [24, 30, 18, 42, 15, 28, 35, 20, 19, 31, 22]
    })

# =========================================================
# 4. SIDEBAR NAVIGATION & SELECTION
# =========================================================
st.sidebar.title("🏫 AVS College Sports AI")
st.sidebar.markdown("---")

selected_sport = st.sidebar.radio("⚽ Select Sport", ["Cricket", "Football"])
selected_team = st.sidebar.radio("👥 Select Team", ["Boys Team", "Girls Team"])

current_squad_df = generate_squad(selected_sport, selected_team.split()[0])
active_player = st.sidebar.selectbox("🎯 Select Active Player to Track:", current_squad_df["Player Name"].tolist())

st.sidebar.markdown("---")
st.sidebar.info("💡 **Camera Instructions:**\nWorks on Laptop Webcams and Mobile Phone Browsers. Allow camera permissions when prompted.")

# =========================================================
# 5. MAIN DASHBOARD UI
# =========================================================
st.title(f"🏆 AVS Engineering College - {selected_sport} ({selected_team})")

tab_squad, tab_vision = st.tabs(["📋 Full 11-Player Squad Roster", "🎥 Live Camera AI Performance Analysis"])

# --- TAB 1: SQUAD ROSTER ---
with tab_squad:
    st.header("Total Squad Overview: 11 Active Field Players")
    
    col1, col2, col3 = st.columns(3)
    top_3 = current_squad_df.head(3)
    
    with col1:
        st.info(f"**#1 {top_3.iloc[0]['Player Name']}**\n\n**Position:** {top_3.iloc[0]['Position']}\n\n**Speed:** {top_3.iloc[0]['Baseline Speed (km/h)']} km/h")
    with col2:
        st.warning(f"**#2 {top_3.iloc[1]['Player Name']}**\n\n**Position:** {top_3.iloc[1]['Position']}\n\n**Speed:** {top_3.iloc[1]['Baseline Speed (km/h)']} km/h")
    with col3:
        st.success(f"**#3 {top_3.iloc[2]['Player Name']}**\n\n**Position:** {top_3.iloc[2]['Position']}\n\n**Speed:** {top_3.iloc[2]['Baseline Speed (km/h)']} km/h")

    st.markdown("### Interactive Team Table")
    st.dataframe(current_squad_df, use_container_width=True, hide_index=True)

# --- TAB 2: LIVE AI CAMERA ---
with tab_vision:
    st.header(f"⚡ Live AI Tracking & Analysis: {active_player}")
    
    col_cam, col_stats = st.columns([2, 1])

    with col_cam:
        camera_file = st.camera_input("Capture Player Action Frame")

    with col_stats:
        st.markdown("### AI Analysis Results")
        live_action = st.empty()
        live_verdict = st.empty()

    if camera_file is not None:
        bytes_data = camera_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        h, w, _ = cv2_img.shape

        try:
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                results = pose.process(rgb_img)

                verdict = "Player Detected - Position Normal"
                action_type = "Idle / Stance"

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(cv2_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    landmarks = results.pose_landmarks.landmark

                    if selected_sport == "Cricket":
                        # Compare Right Wrist relative to Shoulder
                        right_wrist_y = landmarks[16].y
                        right_shoulder_y = landmarks[12].y
                        
                        if right_wrist_y < right_shoulder_y:
                            action_type = "HIGH ARM BOWLING ACTION DETECTED"
                            player_data = current_squad_df[current_squad_df["Player Name"] == active_player].iloc[0]
                            
                            if "Fast" in player_data["Position"] or player_data["Baseline Speed (km/h)"] > 120:
                                verdict = f"🔥 EXCELLENT FAST BOWL DELIVERY! ({player_data['Baseline Speed (km/h)']} km/h)"
                            else:
                                verdict = f"🌀 SPIN / MEDIUM CONTROLLED DELIVERY ({player_data['Baseline Speed (km/h)']} km/h)"
                    
                    elif selected_sport == "Football":
                        right_ankle_x = landmarks[28].x
                        left_ankle_x = landmarks[27].x
                        
                        if abs(right_ankle_x - left_ankle_x) > 0.15:
                            action_type = "POWER KICK / SPRINT EXTENSION DETECTED"
                            player_data = current_squad_df[current_squad_df["Player Name"] == active_player].iloc[0]
                            verdict = f"⚡ HIGH-SPEED SPRINT / STRIKE! Top speed: {player_data['Baseline Speed (km/h)']} km/h"

                    # Draw text overlay on image
                    cv2.rectangle(cv2_img, (10, 10), (w - 10, 60), (0, 0, 0), -1)
                    cv2.putText(cv2_img, f"Tracking: {active_player} ({selected_sport})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    with col_cam:
                        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="AI Keypoint & Pose Mapping", use_container_width=True)

                    with col_stats:
                        live_action.info(f"**Detected Motion:** {action_type}")
                        if "EXCELLENT" in verdict or "HIGH-SPEED" in verdict:
                            live_verdict.success(f"**AI Verdict:**\n\n{verdict}")
                        else:
                            live_verdict.warning(f"**AI Verdict:**\n\n{verdict}")
                else:
                    st.warning("No player pose detected in frame. Ensure full body/arm is visible.")
        except Exception as e:
            st.error(f"AI Detection processing error: {e}")
