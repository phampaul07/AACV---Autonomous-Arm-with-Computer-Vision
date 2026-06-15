import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).resolve().parents[1] / "lib" / "lewansoul-servo-bus" / "src" / "python")
from lewansoul_servo_bus import ServoBus
import time

servo_bus = ServoBus('/dev/cu.usbmodem5C4C1247351')

# Move servo with ID 1 to 90 degrees in 1.0 seconds
servo_bus.move_time_write(1, 120, 1.0)
time.sleep(1.5)  # Wait for the movement to complete




