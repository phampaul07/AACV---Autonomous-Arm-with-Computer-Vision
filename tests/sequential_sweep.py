import sys
import time
import json
from pathlib import Path

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
    1: "center",  # ~129.0°
    2: "max",     # ~233.8°
    3: "min",     # ~14.9°
    4: "min",     # ~22.8°
    5: "center",  # ~116.2°
    6: "max"      # ~178.3°
}

try:
    with open('calibration_results.json', 'r') as f:
        cal_data = json.load(f)
except FileNotFoundError:
    print("Error: Could not find 'calibration_results.json'.")
    sys.exit(1)

def servo_sweep(servo_bus, servo_id, min_ang, center_ang, max_ang, rest_ang):
    servo = servo_bus.get_servo(servo_id)
    
    servo.move_time_write(center_ang, 2.0)
    time.sleep(2.5)
    
    # Sweep to Min
    servo.move_time_write(min_ang, 2.0)
    time.sleep(2.5)
    
    # Sweep to Max
    servo.move_time_write(max_ang, 4.0)
    time.sleep(4.5)

    # Rest Motor in its designated resting position
    servo.move_time_write(rest_ang, 2.0)
    time.sleep(2.5)      

with ServoBus(MAC_PORT, baudrate=1000000, discard_echo=False) as servo_bus:
    print("Starting sequential sweep of all motors...\n")
    
    for servo_id in [1, 2, 3, 4, 5, 6]:
        motor_name = servo_names.get(servo_id, f"Unknown_{servo_id}")
        
        if motor_name in cal_data:
            motor_info = cal_data[motor_name]
            min_ang = motor_info["min_angle"]
            max_ang = motor_info["max_angle"]
            center_ang = motor_info["center_angle"]

            rest_phase = RESTING_STATES.get(servo_id, "center") 
            if rest_phase == "min":
                rest_ang = min_ang
            elif rest_phase == "max":
                rest_ang = max_ang
            else:
                rest_ang = center_ang

            print(f"--- Sweeping {motor_name} (ID {servo_id}) ---")
            print(f"Data: Min {min_ang}° | Center {center_ang}° | Max {max_ang}°")
            print(f"Targeting Resting Phase: {rest_phase.upper()} ({rest_ang}°)")
            
            servo_sweep(servo_bus, servo_id, min_ang, center_ang, max_ang, rest_ang)
            
            print(f"Sweep complete. {motor_name} parked at {rest_phase.upper()}. Taking a 1.5s break...\n")
            time.sleep(1.5) 
            
        else:
            print(f"Skipping {motor_name} (ID {servo_id}) - No calibration data found.\n")
    
    print("All motors swept and parked in their resting positions. Test complete.")