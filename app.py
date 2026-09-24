import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="AthlediX AI - Performance Analytics", page_icon="⚡", layout="wide")

st.title("⚡ AthlediX AI: Sports Performance Analytics Engine")
st.caption("AVS Technovations 2026 | Multi-Sport Athletic Readiness & Workload Optimization")

st.divider()

# --- 1. TRAIN MACHINE LEARNING PERFORMANCE MODEL ---
@st.cache_resource
def train_performance_model():
    np.random.seed(42)
    n = 1000
    acute = np.random.uniform(1000, 4000, n)
    chronic = np.random.uniform(1200, 3000, n)
    acwr = acute / chronic
    sleep = np.random.uniform(4, 10, n)
    fatigue = np.random.randint(1, 11, n)
    soreness = np.random.randint(1, 11, n)
    
    peak_performance = ((acwr >= 0.85) & (acwr <= 1.30) & (sleep >= 7.0) & (fatigue <= 5)).astype(int)
    
    X = pd.DataFrame({'ACWR': acwr, 'Sleep_Hours': sleep, 'Fatigue_Score': fatigue, 'Muscle_Soreness': soreness})
    y = peak_performance
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_performance_model()

# --- 2. MAIN DASHBOARD TABS ---
tab1, tab2 = st.tabs(["🚀 Individual Athlete Analytics", "📊 Squad Performance Matrix"])

# TAB 1: INDIVIDUAL ANALYTICS
with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📋 Select Athlete Profile")
        player_selection = st.selectbox(
            "Select Team Athlete", 
            [
                "Sathish (Soccer - Midfielder)", 
                "Kumar (Cricket - Fast Bowler)", 
                "Sowmiya (Soccer - Forward)", 
                "Nisha (Cricket - All-Rounder)"
            ]
        )
        
        if "Sathish" in player_selection:
            def_acute, def_chronic, def_sleep, def_fatigue, def_soreness = 2400, 2200, 8.0, 3, 2
        elif "Kumar" in player_selection:
            def_acute, def_chronic, def_sleep, def_fatigue, def_soreness = 3500, 2000, 5.0, 9, 8
        elif "Sowmiya" in player_selection:
            def_acute, def_chronic, def_sleep, def_fatigue, def_soreness = 3100, 2000, 6.0, 7, 6
        else:
            def_acute, def_chronic, def_sleep, def_fatigue, def_soreness = 1900, 2200, 8.0, 2, 2

        acute_load = st.number_input("7-Day Acute Training Load (AU)", 500, 5000, def_acute)
        chronic_load = st.number_input("28-Day Chronic Baseline Load (AU)", 500, 5000, def_chronic)
        
        sleep = st.slider("Sleep Duration (Hours)", 3.0, 11.0, def_sleep, 0.5)
        fatigue = st.slider("Fatigue Index (1 = Fresh, 10 = Exhausted)", 1, 10, def_fatigue)
        soreness = st.slider("Muscle Soreness (1 = None, 10 = Severe)", 1, 10, def_soreness)
        
        acwr = round(acute_load / chronic_load, 2)

    with col2:
        st.subheader("📈 Real-Time Performance Predictions")
        
        input_df = pd.DataFrame([[acwr, sleep, fatigue, soreness]], 
                                columns=['ACWR', 'Sleep_Hours', 'Fatigue_Score', 'Muscle_Soreness'])
        
        readiness_score = model.predict_proba(input_df)[0][1] * 100
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Match Readiness", f"{readiness_score:.1f}%")
        m2.metric("Conditioning Base", f"{chronic_load} AU")
        m3.metric("Workload Ratio (ACWR)", f"{acwr}")

        if readiness_score >= 70:
            st.success("🚀 **STATUS: PEAK GAME READINESS**")
            st.write("Athlete is in the optimal performance window. Maximum stamina, sprint speed, and physical output expected.")
        elif 40 <= readiness_score < 70:
            st.warning("⚡ **STATUS: MODERATE PERFORMANCE CAPACITY**")
            st.write("Athlete is carrying accumulated strain. Sub-maximal physical performance expected; monitor training load.")
        else:
            st.error("📉 **STATUS: DEGRADED PERFORMANCE OUTPUT**")
            st.write("High fatigue or acute workload spike detected. Physical capacity and endurance output reduced by an estimated 15–25%.")

        st.markdown("---")
        
        st.subheader("🏋️ 7-Day Fitness vs. Fatigue Balance")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fitness_curve = [chronic_load * 0.9, chronic_load * 0.92, chronic_load * 0.95, chronic_load * 0.98, chronic_load, chronic_load * 1.02, chronic_load * 1.05]
        fatigue_curve = [acute_load * 0.6, acute_load * 0.75, acute_load * 1.1, acute_load * 0.8, acute_load * 0.95, acute_load * 1.2, acute_load]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=fitness_curve, mode='lines+markers', name='Fitness (Baseline)', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=days, y=fatigue_curve, mode='lines+markers', name='Fatigue (Strain)', line=dict(color='red', width=2)))
        
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: SQUAD MATRIX (CSV UPLOAD)
with tab2:
    st.subheader("📁 Automated Squad Roster Analytics")
    uploaded_file = st.file_uploader("Upload Team CSV ('athletes.csv')", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['ACWR'] = (df['Acute_Load_7day'] / df['Chronic_Load_28day']).round(2)
        
        X_squad = df[['ACWR', 'Sleep_Hours', 'Fatigue_Score', 'Muscle_Soreness']]
        df['Readiness_Score (%)'] = (model.predict_proba(X_squad)[:, 1] * 100).round(1)
        df['Performance Status'] = np.where(df['Readiness_Score (%)'] >= 70, '🚀 PEAK READINESS', 
                                   np.where(df['Readiness_Score (%)'] >= 40, '⚡ MODERATE', '🚨 REST REQUIRED'))
        
        st.write("### Squad Optimization Matrix")
        st.dataframe(df[['Player_Name', 'Sport', 'Position', 'ACWR', 'Sleep_Hours', 'Fatigue_Score', 'Readiness_Score (%)', 'Performance Status']], use_container_width=True)
        
        fig_squad = px.bar(df, x='Player_Name', y='Readiness_Score (%)', color='Performance Status', hover_data=['Sport', 'Position'], title="Squad Match Readiness Distribution")
        st.plotly_chart(fig_squad, use_container_width=True)
    else:
        st.info("Upload `athletes.csv` to process squad-wide performance analytics.")