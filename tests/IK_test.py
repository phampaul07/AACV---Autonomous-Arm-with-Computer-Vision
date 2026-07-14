import sys
import time
import math
import json  
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--x', type=float, default=0.0, help="Target X on the board")
parser.add_argument('--y', type=float, default=9, help="Target Y on the board")
args = parser.parse_args()


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

RESTING_STATES = {
    1: "center",  
    2: "max",     
    3: "min",     
    4: "min",     
    5: "center",  
    6: "max"      
}

try:
    with open('calibration_results.json', 'r') as f:
        cal_data = json.load(f)
except FileNotFoundError:
    print("Error: Could not find 'calibration_results.json'.")
    sys.exit(1)


CORRECTIONS_PATH = "arm_corrections.json"


def load_arm_correction_interpolator(path=CORRECTIONS_PATH):

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found. No correction will be applied.")
        return None, None, None, None

    xs = sorted(set(float(item["target"][0]) for item in data))
    ys = sorted(set(float(item["target"][1]) for item in data))

    dx_grid = np.zeros((len(ys), len(xs)))
    dy_grid = np.zeros((len(ys), len(xs)))

    for item in data:
        x, y = item["target"]
        dx, dy = item["delta"]

        x = float(x)
        y = float(y)
        dx = float(dx)
        dy = float(dy)

        x_index = xs.index(x)
        y_index = ys.index(y)

        dx_grid[y_index, x_index] = dx
        dy_grid[y_index, x_index] = dy

    # Interpolators expect input as (y, x), because grid shape is [y][x]
    dx_interp = RegularGridInterpolator(
        (ys, xs),
        dx_grid,
        bounds_error=False,
        fill_value=None
    )

    dy_interp = RegularGridInterpolator(
        (ys, xs),
        dy_grid,
        bounds_error=False,
        fill_value=None
    )

    return dx_interp, dy_interp, xs, ys

def clamp(value, low, high):
    return max(low, min(high, value))

def apply_arm_correction(x_mm, y_mm, dx_interp, dy_interp, xs, ys):

    if dx_interp is None or dy_interp is None:
        return x_mm, y_mm


    x_clamped = clamp(x_mm, min(xs), max(xs))
    y_clamped = clamp(y_mm, min(ys), max(ys))

    # Interpolator input order is (y, x)
    point = np.array([[y_clamped, x_clamped]])

    dx = float(dx_interp(point)[0])
    dy = float(dy_interp(point)[0])

    corrected_x = x_mm + dx
    corrected_y = y_mm + dy

    print("\n--- ARM CORRECTION ---")
    print(f"Raw board target:       x={x_mm:.2f} mm, y={y_mm:.2f} mm")
    print(f"Clamped for lookup:     x={x_clamped:.2f} mm, y={y_clamped:.2f} mm")
    print(f"Correction delta:       dx={dx:+.2f} mm, dy={dy:+.2f} mm")
    print(f"Corrected board target: x={corrected_x:.2f} mm, y={corrected_y:.2f} mm")
    print("----------------------\n")

    return corrected_x, corrected_y

L1 = 115.48 # Length of the upper arm in mm
L2 = 135.81 # Length of the forearm in mm
L3 = 185.71 # Length of the end effector in mm
Z_offset = 127 # Offset in the Z direction in mm
Z_target = 70 # Target Z position in mm
base_offset = 23.11
base_angle_offset = 9

in_x = args.x * 25.4 
in_y = args.y * 25.4

target_x_mm = args.x * 25.4
target_y_mm = args.y * 25.4

dx_interp, dy_interp, xs, ys = load_arm_correction_interpolator()

corrected_x_mm, corrected_y_mm = apply_arm_correction(
    target_x_mm,
    target_y_mm,
    dx_interp,
    dy_interp,
    xs,
    ys
)

LEFT_GRASP_BIAS_MM = -9.5

if target_x_mm < 0:
    corrected_x_mm += LEFT_GRASP_BIAS_MM
    print(f"Applied left-side grasp bias: {LEFT_GRASP_BIAS_MM:+.2f} mm")

robot_x = corrected_x_mm
robot_y = corrected_y_mm + base_offset


IK_result = inverse_kinematics(robot_x, robot_y, L1, L2, L3, Z_offset, Z_target)

if IK_result is None:
    print("The target position is unreachable. Please choose a different position.")
    sys.exit(1)

base_angle, shoulder_angle, elbow_angle, wrist_angle = IK_result

base_cmd = map_angle_to_servo(1, base_angle, cal_data)
base_cmd += base_angle_offset
shoulder_cmd = map_angle_to_servo(2, shoulder_angle, cal_data)
elbow_cmd = map_angle_to_servo(3, elbow_angle, cal_data)
wrist_cmd = map_angle_to_servo(4, wrist_angle, cal_data)


print("\n--- SERVO COMMANDS ---")
print(f"Base servo:     {base_cmd:.2f}°")
print(f"Shoulder servo: {shoulder_cmd:.2f}°")
print(f"Elbow servo:    {elbow_cmd:.2f}°")
print(f"Wrist servo:    {wrist_cmd:.2f}°")
print("----------------------\n")

with ServoBus(MAC_PORT, baudrate=1000000, discard_echo=False) as servo_bus:

    # Move to target position
    open_gripper_angle = cal_data["gripper"]["min_angle"]
    servo_bus.move_time_write(6, open_gripper_angle, 1.5)
    time.sleep(2)
    
    servo_bus.move_time_write(1, base_cmd, 3.0)
    servo_bus.move_time_write(2, shoulder_cmd, 3.0)
    servo_bus.move_time_write(3, elbow_cmd, 3.0)
    servo_bus.move_time_write(4, wrist_cmd, 1.5)


    time.sleep(3.5)

    close_gripper_angle = cal_data["gripper"]["max_angle"]
    servo_bus.move_time_write(6, close_gripper_angle, 1.5)
    time.sleep(2)

    print("Fold to resting position...")
    for servo_id in [1, 2, 3, 4, 5]:
            name = servo_names[servo_id]
            state = RESTING_STATES[servo_id]
            
            rest_angle = cal_data[name][f"{state}_angle"]
            
            servo_bus.move_time_write(servo_id, rest_angle, 2.0)

        # Give the motors time to fold back up before the script ends and cuts communication
    time.sleep(2.5)
    print("Sequence complete.")
