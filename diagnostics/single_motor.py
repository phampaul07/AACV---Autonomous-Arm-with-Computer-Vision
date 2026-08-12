import sys
import time
import json  # ADDED: To save our final data cleanly
from pathlib import Path

# Pathing logic to the -master folder
lib_path = Path(__file__).resolve().parents[1] / "lib" / "lewansoul-servo-bus-master" / "src" / "python"
sys.path.insert(0, str(lib_path))

from lewansoul_servo_bus import ServoBus

MAC_PORT = '/dev/cu.usbmodem5C4C1247351'  


# calibration/ lives next to diagnostics/ at the repo root -- build the
# path from this script's own location instead of assuming a particular cwd.
CALIBRATION_RESULTS_PATH = Path(__file__).resolve().parents[1] / "calibration" / "calibration_results.json"

with open(CALIBRATION_RESULTS_PATH, 'r') as f:
        cal_data = json.load(f)

target_motor = "shoulder_pan"
motor_info = cal_data[target_motor]
motor_id = motor_info["id"]
min_ang = motor_info["min_angle"]
max_ang = motor_info["max_angle"]
center_ang = motor_info["center_angle"]

print(f"Data loaded for {target_motor} (ID {motor_id}):")
print(f"  Min: {min_ang}° | Center: {center_ang}° | Max: {max_ang}°\n")

with ServoBus(MAC_PORT, baudrate=1000000, discard_echo=False) as servo_bus:
    
    servo_bus.move_time_write(1, center_ang, 2.0)
    time.sleep(2.5)
    servo_bus.move_time_write(1, min_ang, 2.0)
    time.sleep(2.5)
    servo_bus.move_time_write(1, max_ang, 2.0)
    time.sleep(2.5)
    servo_bus.move_time_write(1, center_ang, 2.0)
    time.sleep(2.5)



