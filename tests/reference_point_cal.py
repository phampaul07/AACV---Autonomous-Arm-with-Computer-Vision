import sys
import time
import json  
from pathlib import Path

# Add your paths back in so python can find your custom libraries
lib_path = Path(__file__).resolve().parents[1] / "core" 
sys.path.insert(0, str(lib_path))
from IK_solver import map_angle_to_servo

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

# Convert your desired Math Angles to Hardware Angles using your JSON!
shoulder_cmd = map_angle_to_servo(2, 90, cal_data)
elbow_cmd = map_angle_to_servo(3, 180, cal_data)
wrist_cmd = map_angle_to_servo(4, 0, cal_data)

print(f"Mapped Shoulder: {shoulder_cmd:.2f}°")
print(f"Mapped Elbow: {elbow_cmd:.2f}°")
print(f"Mapped Wrist: {wrist_cmd:.2f}°")

with ServoBus(MAC_PORT, baudrate=1000000, discard_echo=False) as servo_bus:

    # Move to the converted hardware angles safely
    print("Moving to straight-up position...")
    servo_bus.move_time_write(2, shoulder_cmd, 3.0)
    servo_bus.move_time_write(3, elbow_cmd, 3.0)
    servo_bus.move_time_write(4, wrist_cmd, 3.0)

    # Hold the pose so you can look at the physical arm and measure it
    time.sleep(10)

    print("Fold to resting position...")
    for servo_id in [1, 2, 3, 4, 5]:
        name = servo_names[servo_id]
        state = RESTING_STATES[servo_id]
        
        rest_angle = cal_data[name][f"{state}_angle"]
        
        servo_bus.move_time_write(servo_id, rest_angle, 2.0)

    # Give the motors time to fold back up before the script ends
    time.sleep(2.5)
    print("Sequence complete.")