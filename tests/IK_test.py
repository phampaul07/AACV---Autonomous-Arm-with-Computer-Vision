import sys
import time
import math
import json  
from pathlib import Path

lib_path = Path(__file__).resolve().parents[1] / "core" 
sys.path.insert(0, str(lib_path))
from IK_solver import inverse_kinematics, map_angle_to_servo

lib_path = Path(__file__).resolve().parents[1] / "lib" / "lewansoul-servo-bus-master" / "src" / "python"
sys.path.insert(0, str(lib_path))

from lewansoul_servo_bus import ServoBus

MAC_PORT = '/dev/cu.usbmodem5C4C1247351'  

servo_names = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper"
}

try:
    with open('calibration_results.json', 'r') as f:
        cal_data = json.load(f)
except FileNotFoundError:
    print("Error: Could not find 'calibration_results.json'.")
    sys.exit(1)

L1 = 116 # Length of the upper arm in mm
L2 = 135 # Length of the forearm in mm
L3 = 167 # Length of the end effector in mm
Z_offset = 118 # Offset in the Z direction in mm
Z_target = 22.5 # Target Z position in mm

x = 100  # X position in mm
y = 300  # Y position in mm

IK_result = inverse_kinematics(x, y, L1, L2, L3, Z_offset, Z_target)

if IK_result is None:
    print("The target position is unreachable. Please choose a different position.")
    sys.exit(1)

base_angle, shoulder_angle, elbow_angle, wrist_angle = IK_result

base_cmd = map_angle_to_servo(1, base_angle, cal_data, math_center_offset=90, inverted=True)
shoulder_cmd = map_angle_to_servo(2, shoulder_angle, cal_data, math_center_offset=0, inverted=True)
elbow_cmd = map_angle_to_servo(3, elbow_angle, cal_data, math_center_offset=180, inverted=True)
wrist_cmd = map_angle_to_servo(4, wrist_angle, cal_data, math_center_offset=0, inverted=True)

with ServoBus(MAC_PORT, baudrate=1000000, discard_echo=False) as servo_bus:

    servo_bus.move_time_write(1, base_cmd, 2.0)
    time.sleep(2.5)
    servo_bus.move_time_write(2, shoulder_cmd, 2.0)
    time.sleep(2.5)
    servo_bus.move_time_write(3, elbow_cmd, 2.0)
    time.sleep(2.5)
    servo_bus.move_time_write(4, wrist_cmd, 2.0)
    time.sleep(2.5)

    servo_bus.move_time_write(1, base_cmd, 2.0)







