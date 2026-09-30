import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import sqlite3
import subprocess
from ultralytics import YOLO

# Safe MediaPipe Import
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None

st.set_page_config(page_title="AthlediX AI Engine", layout="wide")
st.title("🏆 AthlediX AI: Direct YOLO11 Local Engine")

# Load YOLO11 Model (e.g., standard object detection model)
@st.cache_resource
def load_yolo():
    return YOLO("yolo11n.pt")  # Nano version for speed

yolo_model = load_yolo()

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    if st.button("🚀 Process Video with YOLO11"):
        cap = cv2.VideoCapture(tfile.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO11 direct inference
            results = yolo_model(frame, verbose=False)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)

        cap.release()
        out.release()
        st.success("YOLO11 Processing Complete!")
        
        with open(out_path, 'rb') as vf:
            st.video(vf.read())
