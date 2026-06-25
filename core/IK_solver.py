import json
import math
import sys 

try:
    with open('calibration_results.json', 'r') as f:
        cal_data = json.load(f)
except FileNotFoundError:
    print("Error: Could not find 'calibration_results.json'.")
    sys.exit(1)

def inverse_kinematics(x, y, L1, L2, L3, Z_offset, Z_target):
    #Measurement units in mm 
    Z_w = Z_target - Z_offset  # Z position of the wrist joint  # X position of the wrist joint (accounting for the length of the end effector)
    R = math.sqrt(x**2 + y**2)  # Distance from the origin to the point (x, y)
    R_w = R - L3  # X position of the wrist joint (accounting for the length of the end effector)
    D = math.sqrt(R_w**2 + Z_w**2)  # Distance from the shoulder joint to the wrist joint

    max_reach = L1 + L2 + L3
    if D > max_reach:
        print("Target position is out of reach. Please choose a different position.")
        return None
    min_reach = 0
    if D < abs(min_reach):
        print("Target position is too close to the base. Please choose a different position.")
        return None

    base_angle = math.degrees(math.atan2(y, x))  # Base angle to reach the point (x, y) 

    shoulder_angle = math.degrees(math.acos((D**2 + L1**2 - L2**2) / (2 * D * L1)) + math.atan2(Z_w, R_w))  # Angle of the shoulder joint
    elbow_angle = math.degrees(math.acos((L1**2 + L2**2 - D**2) / (2 * L1 * L2)))  # Angle of the elbow joint
    wrist_angle = 180 - (shoulder_angle + elbow_angle)  # Angle of the wrist joint -- always horizontal to the ground 

    return base_angle, shoulder_angle, elbow_angle, wrist_angle

def map_angle_to_servo(servo_id, math_angle, cal_data, math_center_offset=0, inverted=False):
    servo_data = None
    for name, data in cal_data.items():
        if data["id"] == servo_id:
            servo_data = data
            break
            
    if servo_data is None:
        return math_angle # Fallback if not found
        
    center = servo_data["center_angle"]
    movement = math_angle - math_center_offset
    
    if inverted:
        hardware_angle = center - movement
    else:
        hardware_angle = center + movement
        
    if hardware_angle < servo_data["min_angle"]: hardware_angle = servo_data["min_angle"]
    if hardware_angle > servo_data["max_angle"]: hardware_angle = servo_data["max_angle"]
        
    return hardware_angle


