import sys
import time
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

print("Press Ctrl+C in the terminal to stop the scan.")


print("\n" * 7)

try:
    with ServoBus(MAC_PORT, baudrate=1000000, discard_echo=False) as servo_bus:
        
        while True:
            sys.stdout.write("\033[7F")
            
            print("--- Scanning IDs 1-6 ---          ")
            
            for servo_id in [1, 2, 3, 4, 5, 6]:
                try:
                    servo = servo_bus.get_servo(servo_id)
                    current_pos = servo.pos_read()
                    
                    if current_pos is not None:
                        print(f"ID {servo_id} ({servo_names.get(servo_id, 'Unknown')}): {current_pos}°                ")
                    else:
                        print(f"ID {servo_id} ({servo_names.get(servo_id, 'Unknown')}): No response                 ")
                        
                except Exception as e:
                    print(f" Error ID {servo_id}                        ")
                    
                time.sleep(0.01)
                
            time.sleep(0.01) 

except KeyboardInterrupt:
    print("\nScan stopped by user.")
    print("Test complete.")