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

max_angles = {i: 0.0 for i in range(1, 7)}
min_angles = {i: 360.0 for i in range(1, 7)}

print("Opening connection (1M Baud, No Echo)...")
print("Press Ctrl+C in the terminal to stop the scan.")
print("\n" * 7)

try:
    with ServoBus(MAC_PORT, baudrate=1000000, discard_echo=False) as servo_bus:
        
        while True:
            sys.stdout.write("\033[7F")
            print("--- Calibration Scanner: Move the arm to find limits ---      ")
            
            for servo_id in [1, 2, 3, 4, 5, 6]:
                name = servo_names.get(servo_id, "Unknown")
                
                try:
                    servo = servo_bus.get_servo(servo_id)
                    pos = servo.pos_read()
                    
                    if pos is not None:
                        max_angles[servo_id] = max(max_angles[servo_id], pos)
                        min_angles[servo_id] = min(min_angles[servo_id], pos)
                        
                        center = (max_angles[servo_id] + min_angles[servo_id]) / 2.0
                        name = servo_names.get(servo_id, 'Unknown')
                        
                        print(f"ID {servo_id} ({name:<14}) | Min: {min_angles[servo_id]:>5.1f}° | Center: {center:>5.1f}° | Max: {max_angles[servo_id]:>5.1f}°      ")
                    else:
                        print(f"ID {servo_id} ({servo_names.get(servo_id, 'Unknown')}): No response                 ")
                        
                except Exception as e:
                    print(f"Error {name:<14} (ID {servo_id})                                                   ")
                    

                time.sleep(0.02)
                
            time.sleep(0.1) 

except KeyboardInterrupt:
    print("\n\nScan stopped by user.")
    print("Calibration complete. Writing results to 'calibration_results.json'...")
    
    final_data = {}
    
    for servo_id in range(1, 7):
        name = servo_names.get(servo_id, 'Unknown')
        min_ang = min_angles[servo_id]
        max_ang = max_angles[servo_id]
        
        if min_ang < 360.0 or max_ang > 0.0:
            center = (max_ang + min_ang) / 2.0
            
            final_data[name] = {
                "id": servo_id,
                "min_angle": round(min_ang, 1),
                "center_angle": round(center, 1),
                "max_angle": round(max_ang, 1)
            }

    with open('calibration_results.json', 'w') as f:
        json.dump(final_data, f, indent=4)
        
    print("Results saved.")