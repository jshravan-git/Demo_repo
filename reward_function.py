import math

# Use a global variable to store the previous steering angle for smoothness reward
prev_steering_angle = 0

def reward_function(params):
    global prev_steering_angle
    
    # Read input parameters
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    steering_angle = params['steering_angle']
    speed = params['speed']
    all_wheels_on_track = params['all_wheels_on_track']
    is_offtrack = params['is_offtrack']
    progress = params['progress']
    steps = params['steps']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    
    # Initial reward
    reward = 1.0
    
    # 1. Strong penalty for going off-track or losing control
    if is_offtrack or not all_wheels_on_track:
        prev_steering_angle = 0
        return 1e-3
    
    # 2. Symmetrical reward for staying on the track (using markers)
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    if distance_from_center <= marker_1:
        reward += 1.0
    elif distance_from_center <= marker_2:
        reward += 0.5
    elif distance_from_center <= marker_3:
        reward += 0.1
    else:
        reward = 1e-3

    # 3. Heading alignment
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = math.degrees(math.atan2(next_point[1] - prev_point[1],
                                              next_point[0] - prev_point[0]))
    
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    
    DIRECTION_THRESHOLD = 10.0
    if direction_diff < DIRECTION_THRESHOLD:
        reward += 1.0
    else:
        reward *= 0.5

    # 4. Proactive curve detection and speed/steering control
    NUM_POINTS_FOR_CURVE = 5
    future_waypoints = waypoints[closest_waypoints[1]:closest_waypoints[1] + NUM_POINTS_FOR_CURVE]
    
    is_on_curve = False
    if len(future_waypoints) >= NUM_POINTS_FOR_CURVE:
        track_directions = []
        for i in range(1, len(future_waypoints)):
            prev_point_seg = future_waypoints[i - 1]
            next_point_seg = future_waypoints[i]
            track_direction_segment = math.degrees(math.atan2(next_point_seg[1] - prev_point_seg[1],
                                                              next_point_seg[0] - prev_point_seg[0]))
            track_directions.append(track_direction_segment)
            
        direction_changes = [abs(track_directions[i] - track_directions[i-1]) for i in range(1, len(track_directions))]
        average_direction_change = sum(direction_changes) / len(direction_changes)
        
        CURVE_THRESHOLD = 15.0
        if average_direction_change > CURVE_THRESHOLD:
            is_on_curve = True

    SPEED_STRAIGHT = 3.5
    SPEED_CURVE = 2.0
    
    if is_on_curve:
        if speed < SPEED_CURVE:
            reward += 1.5
        else:
            reward *= 0.8
        
        if abs(steering_angle) > 15:
            reward += 1.0
    else:
        if speed >= SPEED_STRAIGHT:
            reward += 2.0
        
        if abs(steering_angle) < 5.0:
            reward += 1.0
        else:
            reward *= 0.5 # Penalize zig-zag on straights
    
    # 5. Smoothness reward
    steering_change = abs(steering_angle - prev_steering_angle)
    if steering_change < 5.0:
        reward += 0.5
    else:
        reward *= 0.75

    # 6. Progress and completion bonus
    if progress > 99.0:
        reward += 50.0
    else:
        if steps > 0:
            reward += progress / 100.0

    # Update previous steering angle for the next step
    prev_steering_angle = steering_angle
    
    return float(max(reward, 1e-3))
