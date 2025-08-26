import math

def reward_function(params):
    # Read input parameters
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    steering = abs(params['steering_angle'])
    speed = params['speed']
    all_wheels_on_track = params['all_wheels_on_track']
    is_offtrack = params['is_offtrack']
    progress = params['progress']
    steps = params['steps']
    
    # Initial reward
    reward = 1.0
    
    # Immediate disqualification if off track
    if is_offtrack or not all_wheels_on_track:
        return 1e-3  # very low reward
    
    # Track direction calculation
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = math.atan2(next_point[1] - prev_point[1],
                                  next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction)
    
    # Difference between track direction and heading
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    
    # Direction alignment check
    DIRECTION_THRESHOLD = 10.0
    if direction_diff > DIRECTION_THRESHOLD:
        reward *= 0.5  # penalize if not aligned
    else:
        reward += 1.0  # reward for good alignment
    
    # Curve detection logic
    curve_angle = abs(track_direction - heading)
    if curve_angle > 15.0:  # curve detected
        # Encourage alignment
        if direction_diff < 8.0:
            reward += 2.0
        
        # Encourage steering into curve
        if steering < 10:
            reward *= 0.5  # penalize for going straight
        else:
            reward += 1.0  # reward proper steering
        
        # Penalize overspeed
        if speed > 2.0:
            reward *= 0.7
    else:  # straight section (non-curve)
        # Encourage higher speed on straight
        if speed >= 3.0:  # you can adjust the threshold
            reward += 2.0  # strong reward for high speed
        elif speed >= 2.0:
            reward += 1.0  # moderate reward for decent speed
    
    # Bonus for completing laps
    if progress > 99.0:
        reward += 50.0
    
    # Reward based on steps (efficiency)
    if steps > 0:
        reward += progress / steps  # encourage efficiency
    
    return float(reward)

