import sys
import time
import json
from pathlib import Path


lib_path = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(lib_path))
from IK_solver import inverse_kinematics, map_angle_to_servo

lib_path = (
    Path(__file__).resolve().parents[1]
    / "lib"
    / "lewansoul-servo-bus-master"
    / "src"
    / "python"
)
sys.path.insert(0, str(lib_path))

from lewansoul_servo_bus import ServoBus


MAC_PORT = "/dev/cu.usbmodem5C4C1247351"

INCH_TO_MM = 25.4
CORRECTIONS_PATH = "arm_corrections.json"

L1 = 115.48
L2 = 135.81
L3 = 177.51
Z_offset = 127
Z_target = 80
base_offset = 23.11
base_angle_offset = 5


# -----------------------------
# 3x3 calibration grid
# 9 total points
# -----------------------------
TEST_GRID_INCHES = [
    (-5, 13), (0, 13), (5, 13),
    (-5, 11), (0, 11), (5, 11),
    (-5, 9),  (0, 9),  (5, 9),
]


servo_names = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}

RESTING_STATES = {
    1: "center",
    2: "max",
    3: "min",
    4: "min",
    5: "center",
    6: "max",
}


# -----------------------------
# JSON helpers
# -----------------------------
def load_calibration():
    try:
        with open("calibration_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: Could not find calibration_results.json")
        sys.exit(1)


def load_corrections():
    try:
        with open(CORRECTIONS_PATH, "r") as f:
            data = json.load(f)

        corrections = {}
        for item in data:
            target = tuple(item["target"])
            delta = tuple(item["delta"])
            corrections[target] = delta

        return corrections

    except FileNotFoundError:
        return {}


def save_corrections(corrections):
    data = []

    for target, delta in corrections.items():
        data.append(
            {
                "target": list(target),
                "delta": list(delta),
            }
        )

    with open(CORRECTIONS_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved {len(data)} correction points to {CORRECTIONS_PATH}")


# -----------------------------
# Movement helpers
# -----------------------------
def move_to_board_point(servo_bus, cal_data, board_x_mm, board_y_mm):
    """
    board_x_mm, board_y_mm are in board coordinates.
    Origin: bottom-center of grid board.
    +x = right
    +y = forward/up board
    """

    robot_x = board_x_mm
    robot_y = board_y_mm + base_offset

    ik_result = inverse_kinematics(
        robot_x,
        robot_y,
        L1,
        L2,
        L3,
        Z_offset,
        Z_target,
    )

    if ik_result is None:
        print("Target is unreachable.")
        return False

    base_angle, shoulder_angle, elbow_angle, wrist_angle = ik_result

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

    servo_bus.move_time_write(1, base_cmd, 3.0)
    servo_bus.move_time_write(2, shoulder_cmd, 3.0)
    servo_bus.move_time_write(3, elbow_cmd, 3.0)
    servo_bus.move_time_write(4, wrist_cmd, 1.5)

    # Keep gripper open while measuring.
    open_gripper_angle = cal_data["gripper"]["min_angle"]
    servo_bus.move_time_write(6, open_gripper_angle, 1.5)

    time.sleep(3.5)
    return True


def rest_arm(servo_bus, cal_data):
    print("\nFolding to resting position...")

    for servo_id in [1, 2, 3, 4, 5]:
        name = servo_names[servo_id]
        state = RESTING_STATES[servo_id]
        rest_angle = cal_data[name][f"{state}_angle"]
        servo_bus.move_time_write(servo_id, rest_angle, 2.0)

    time.sleep(2.5)
    print("Rest complete.")


# -----------------------------
# Input helpers
# -----------------------------
def get_float_or_command(prompt):
    value = input(prompt).strip().lower()

    if value in ["s", "skip"]:
        return "skip"

    if value in ["q", "quit"]:
        return "quit"

    try:
        return float(value)
    except ValueError:
        print("Invalid input. Type a number, 's', or 'q'.")
        return get_float_or_command(prompt)


# -----------------------------
# Main calibration loop
# -----------------------------
def main():
    cal_data = load_calibration()
    corrections = load_corrections()

    print("\nManual Arm Calibration")
    print("----------------------")
    print("This will create a 3x3 correction map.")
    print("The arm will move to 9 known grid points.")
    print("For each point, measure where the claw actually landed.")
    print("Enter the actual landing coordinate in inches.")
    print("After each point, the arm will return to rest.")
    print("\nCommands:")
    print("  s = skip point")
    print("  q = quit calibration")

    print("\nCalibration points:")
    for i, (x_in, y_in) in enumerate(TEST_GRID_INCHES, start=1):
        print(f"  {i}. x={x_in:>5.2f} in, y={y_in:>5.2f} in")

    input("\nPress ENTER to connect and begin...")

    with ServoBus(MAC_PORT, baudrate=1000000, discard_echo=False) as servo_bus:
        # Start from rest for consistency.
        rest_arm(servo_bus, cal_data)

        for index, (target_x_in, target_y_in) in enumerate(TEST_GRID_INCHES, start=1):
            target_x_mm = target_x_in * INCH_TO_MM
            target_y_mm = target_y_in * INCH_TO_MM

            print("\n" + "=" * 60)
            print(f"Point {index}/{len(TEST_GRID_INCHES)}")
            print(f"Target: x={target_x_in:.2f} in, y={target_y_in:.2f} in")
            print(f"Target: x={target_x_mm:.2f} mm, y={target_y_mm:.2f} mm")
            print("=" * 60)

            command = input("Press ENTER to move, 's' to skip, or 'q' to quit: ").strip().lower()

            if command in ["q", "quit"]:
                break

            if command in ["s", "skip"]:
                print("Skipped.")
                continue

            moved = move_to_board_point(
                servo_bus,
                cal_data,
                target_x_mm,
                target_y_mm,
            )

            if not moved:
                print("Skipping unreachable point.")
                continue

            print("\nMeasure where the claw actually landed on the grid board.")
            print("Use the SAME coordinate system as the target:")
            print("  +x = right")
            print("  -x = left")
            print("  +y = forward/up board")
            print("\nExample:")
            print("  If target was x=-5, y=13")
            print("  and claw landed at x=-4.8, y=13.2")
            print("  enter actual x = -4.8, actual y = 13.2")

            actual_x_in = get_float_or_command("\nActual x in inches: ")

            if actual_x_in == "quit":
                break

            if actual_x_in == "skip":
                print("Skipped.")
                input("\nPress ENTER to return to rest...")
                rest_arm(servo_bus, cal_data)
                continue

            actual_y_in = get_float_or_command("Actual y in inches: ")

            if actual_y_in == "quit":
                break

            if actual_y_in == "skip":
                print("Skipped.")
                input("\nPress ENTER to return to rest...")
                rest_arm(servo_bus, cal_data)
                continue

            actual_x_mm = actual_x_in * INCH_TO_MM
            actual_y_mm = actual_y_in * INCH_TO_MM

            dx = target_x_mm - actual_x_mm
            dy = target_y_mm - actual_y_mm

            corrections[(target_x_mm, target_y_mm)] = (dx, dy)

            print("\nCorrection saved:")
            print(f"Target: x={target_x_mm:.2f} mm, y={target_y_mm:.2f} mm")
            print(f"Actual: x={actual_x_mm:.2f} mm, y={actual_y_mm:.2f} mm")
            print(f"Delta:  dx={dx:+.2f} mm, dy={dy:+.2f} mm")

            corrected_command_x = target_x_mm + dx
            corrected_command_y = target_y_mm + dy

            print("\nMeaning:")
            print(
                f"Next time the robot wants this point, "
                f"it should command x={corrected_command_x:.2f} mm, "
                f"y={corrected_command_y:.2f} mm before IK."
            )

            save_corrections(corrections)

            input("\nPress ENTER to return to rest...")
            rest_arm(servo_bus, cal_data)

            input("\nPress ENTER for next point...")

        print("\nCalibration loop ended.")
        rest_arm(servo_bus, cal_data)

    print("\nManual calibration complete.")
    print(f"Corrections saved in {CORRECTIONS_PATH}")


if __name__ == "__main__":
    main()