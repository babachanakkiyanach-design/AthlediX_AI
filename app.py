if st.button("🚀 Run Roboflow Workflow"):
            with st.spinner("Executing Roboflow Serverless Workflow & MediaPipe..."):
                cap = cv2.VideoCapture(input_video_path)
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0 or np.isnan(fps):
                    fps = 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Temporary file for raw OpenCV output
                temp_raw_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_raw_video, fourcc, fps, (width, height))

                bowler_3d_angles = defaultdict(list)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 1. Query Roboflow Serverless Workflow
                    workflow_output = process_roboflow_workflow_frame(frame)

                    # 2. Calculate MediaPipe Pose Angle
                    angle_3d = 160
                    if mp_pose_engine is not None:
                        mp_results = mp_pose_engine.process(rgb_frame)
                        if mp_results.pose_world_landmarks:
                            lm = mp_results.pose_world_landmarks.landmark
                            angle_3d = calculate_3d_angle(lm[12], lm[14], lm[16])

                    # 3. Parse and Draw Workflow Output Predictions
                    if workflow_output and isinstance(workflow_output, list) and len(workflow_output) > 0:
                        predictions = workflow_output[0].get("predictions", [])
                        for idx, pred in enumerate(predictions):
                            if isinstance(pred, dict):
                                x, y = pred.get("x", 0), pred.get("y", 0)
                                w, h = pred.get("width", 0), pred.get("height", 0)
                                t_id = pred.get("tracker_id", pred.get("detection_id", "1"))
                                cls_name = pred.get("class", "object")
                                conf = pred.get("confidence", 0.0)

                                x1, y1 = int(x - w / 2), int(y - h / 2)
                                x2, y2 = int(x + w / 2), int(y + h / 2)

                                bowler_3d_angles[t_id].append(angle_3d)

                                color = BOWLER_COLORS[idx % len(BOWLER_COLORS)]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(
                                    frame,
                                    f"ID:{t_id} {cls_name} ({conf:.2f}) | Arm: {angle_3d}deg",
                                    (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                                )

                    # Overlay Title
                    cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
                    cv2.putText(frame, "AthlediX AI: Roboflow Workflow + MediaPipe 3D Engine",
                                (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    out.write(frame)

                cap.release()
                out.release()

                # Convert raw OpenCV video to web-playable H.264 video
                final_playable_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                convert_to_h264(temp_raw_video, final_playable_video)

                st.subheader("🎬 AI Processed Output Video")
                
                # Stream binary buffer to ensure video plays cleanly in web browsers
                with open(final_playable_video, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
