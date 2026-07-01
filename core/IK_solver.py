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
    deadzone = 226.31


    print(f"--- DEBUG ---")
    print(f"Target R: {R:.2f} mm | Wrist R_w: {R_w:.2f} mm | Hypotenuse D: {D:.2f} mm")
    print(f"Max Wrist Reach (L1+L2): {L1+L2:.2f} mm")
    print(f"-------------")


    
    if R < deadzone:
        print("Target position is too close to the base. Please choose a different position.")
        return None

    base_angle = math.degrees(math.atan2(y, x))  # Base angle to reach the point (x, y) 

    shoulder_angle = math.degrees(math.acos((D**2 + L1**2 - L2**2) / (2 * D * L1)) + math.atan2(Z_w, R_w))  # Angle of the shoulder joint
    elbow_angle = math.degrees(math.acos((L1**2 + L2**2 - D**2) / (2 * L1 * L2)))  # Angle of the elbow joint
    wrist_angle = 180 - (shoulder_angle + elbow_angle)  # Angle of the wrist joint -- always horizontal to the ground 

    print(f"--- MATH ANGLES ---")
    print(f"Base: {base_angle:.2f} | Shoulder: {shoulder_angle:.2f} | Elbow: {elbow_angle:.2f} | Wrist: {wrist_angle:.2f}")

    return base_angle, shoulder_angle, elbow_angle, wrist_angle

    

def map_angle_to_servo(servo_id, joint_angle, cal_data, clamp=True):
    servo_data = None
    servo_name = None

    for name, data in cal_data.items():
        if data["id"] == servo_id:
            servo_data = data
            servo_name = name
            break

    if servo_data is None:
        raise ValueError(f"No calibration found for servo ID {servo_id}")


    if "reference_joint_angle" not in servo_data:
        raise ValueError(
            f"{servo_name} does not have IK mapping fields. "
            f"Use fixed_angle/open_angle/close_angle instead."
        )

    reference_joint = servo_data["reference_joint_angle"]
    reference_servo = servo_data["reference_servo_angle"]
    scale = servo_data.get("servo_degrees_per_joint_degree", 1.0)
    inverted = servo_data.get("inverted", False)

    sign = -1 if inverted else 1

    hardware_angle = reference_servo + sign * (
        joint_angle - reference_joint
    ) * scale

    min_angle = servo_data["min_angle"]
    max_angle = servo_data["max_angle"]

    if hardware_angle < min_angle or hardware_angle > max_angle:
        msg = (
            f"{servo_name} requested {hardware_angle:.2f}°, "
            f"outside safe range [{min_angle:.2f}, {max_angle:.2f}]"
        )

        if clamp:
            print("Warning:", msg)
            hardware_angle = max(min_angle, min(max_angle, hardware_angle))
        else:
            raise ValueError(msg)

    return hardware_angle