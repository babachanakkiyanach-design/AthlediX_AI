# Send frame through YOLO11 Tracking/Detection
try:
    # Attempt ByteTrack/BoT-SORT object tracking
    results = yolo_model.track(frame, conf=conf_thresh, persist=True, verbose=False, classes=[0])[0]
    has_tracking = results.boxes is not None and results.boxes.id is not None
except Exception as tracking_error:
    # Fallback to standard detection if tracking/lap module fails
    results = yolo_model.predict(frame, conf=conf_thresh, verbose=False, classes=[0])[0]
    has_tracking = False

if results.boxes is not None:
    boxes = results.boxes.xyxy.cpu().numpy()
    # Check if tracking IDs are present
    track_ids = results.boxes.id.int().cpu().numpy() if has_tracking else list(range(len(boxes)))

    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        current_pos = (center_x, center_y)

        # Calculate Speed
        speed_kmh = 0.0
        if has_tracking and track_id in previous_positions:
            prev_pos = previous_positions[track_id]
            speed_kmh = compute_speed_kmh(prev_pos, current_pos, time_per_processed_frame, pixel_to_meter)

        if has_tracking:
            previous_positions[track_id] = current_pos

        # Record History Log
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

        # Draw Annotations
        label = f"ID:{track_id} | {speed_kmh:.1f} km/h" if has_tracking else f"Player | {conf_thresh:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
