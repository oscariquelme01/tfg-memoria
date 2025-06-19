def get_goalkeeper_view(all_detections, previous_yaw=None, previous_pitch=None, 
                       frame=None, perspective_width=1920, perspective_height=1080,
                       camera_yaw=0, camera_pitch=0, camera_fov=90,
                       smoothing_factor=0.7, max_yaw_change=15, max_pitch_change=10):
    if not len(all_detections):
        return None
    
    # Separate persons and balls
    persons = [d for d in all_detections if d.get('class', 'person') == 'person']
    balls = [d for d in all_detections if d.get('class', '') == 'ball']
    
    if not persons:
        return None
    
    # Find the best goalkeeper candidate
    best_goalkeeper = persons[0]
    best_score = 0
    
    for detection in persons:
        # Calculate composite score (90% distance, 10% confidence)
        distance_score = (1 / detection['distance']) * 0.9
        confidence_score = detection['conf'] * 0.1
        total_score = distance_score + confidence_score
        
        if total_score > best_score:
            best_score = total_score
            best_goalkeeper = detection
    
    # Calculate base yaw and pitch from goalkeeper position
    gk_bbox = best_goalkeeper['bbox']
    base_yaw, base_pitch = yolo_box_to_yaw_pitch(
        gk_bbox[0], gk_bbox[1], gk_bbox[2], gk_bbox[3],
        perspective_width, perspective_height,
        camera_yaw, camera_pitch, camera_fov
    )
    
    # Ball influence on camera positioning
    target_yaw, target_pitch = base_yaw, base_pitch
    
    if balls:
        # Find the closest/most confident ball
        best_ball = max(balls, key=lambda b: (1/b['distance']) * 0.8 + b['conf'] * 0.2)
        
        ball_bbox = best_ball['bbox']
        ball_yaw, ball_pitch = yolo_box_to_yaw_pitch(
            ball_bbox[0], ball_bbox[1], ball_bbox[2], ball_bbox[3],
            perspective_width, perspective_height,
            camera_yaw, camera_pitch, camera_fov
        )
        
        # Blend goalkeeper and ball positions
        # Weight based on ball distance and goalkeeper-ball proximity
        ball_weight = min(0.4, 1.0 / (best_ball['distance'] + 1))  # Max 40% influence
        
        # If ball is very close to goalkeeper, reduce ball influence
        angle_diff = math.sqrt((ball_yaw - base_yaw)**2 + (ball_pitch - base_pitch)**2)
        if angle_diff < 20:  # If ball is within 20 degrees of goalkeeper
            ball_weight *= 0.5
        
        target_yaw = base_yaw * (1 - ball_weight) + ball_yaw * ball_weight
        target_pitch = base_pitch * (1 - ball_weight) + ball_pitch * ball_weight
    
    # Apply smoothing if previous values exist
    if previous_yaw is not None and previous_pitch is not None:
        # Calculate the change from previous frame
        yaw_change = target_yaw - previous_yaw
        pitch_change = target_pitch - previous_pitch
        
        # Limit maximum change per frame to prevent jarring movements
        yaw_change = max(-max_yaw_change, min(max_yaw_change, yaw_change))
        pitch_change = max(-max_pitch_change, min(max_pitch_change, pitch_change))
        
        # Apply smoothing using exponential moving average
        smoothed_yaw = previous_yaw + yaw_change * (1 - smoothing_factor)
        smoothed_pitch = previous_pitch + pitch_change * (1 - smoothing_factor)
        
        target_yaw = smoothed_yaw
        target_pitch = smoothed_pitch
    
    # Generate the remapped frame if frame is provided
    remapped_frame = None
    if frame is not None:
        remapped_frame = remap_single_view(frame, target_yaw, target_pitch, camera_fov)
    
    return {
        'detection': best_goalkeeper,
        'yaw': target_yaw,
        'pitch': target_pitch,
        'frame': remapped_frame,
        'ball_detected': len(balls) > 0,
        'ball_influence': ball_weight if balls else 0.0
    }
