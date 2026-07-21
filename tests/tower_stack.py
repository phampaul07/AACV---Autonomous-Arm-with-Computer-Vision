import sys
import time
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# -----------------------------
# Path setup
# -----------------------------
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


# -----------------------------
# Same config as IK_test.py
# -----------------------------
MAC_PORT = "/dev/cu.usbmodem5C4C1247351"

CORRECTIONS_PATH = "arm_corrections.json"

L1 = 115.48
L2 = 135.81
L3 = 185.71

Z_offset = 127
Z_target = 70

base_offset = 23.11
base_angle_offset = 9


# -----------------------------
# Cube / stack settings
# -----------------------------
BLOCK_HEIGHT_MM = 30

# This is ONLY for pickup/grabbing.
# It helps the left side of the claw clear the block.
LEFT_PICKUP_GRASP_BIAS_MM = -9.5

# These are ONLY for placement.
# Start at 0.0. Adjust later only if placement is still slightly off.
PLACE_X_BIAS_MM = 20.32
PLACE_Y_BIAS_MM = -33.02


# -----------------------------
# Servo names / rest state
# -----------------------------
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
# Load calibration
# -----------------------------
try:
    with open("calibration_results.json", "r") as f:
        cal_data = json.load(f)
except FileNotFoundError:
    print("Error: Could not find 'calibration_results.json'.")
    sys.exit(1)


# -----------------------------
# Correction map
# -----------------------------
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

    dx_interp = RegularGridInterpolator(
        (ys, xs),
        dx_grid,
        bounds_error=False,
        fill_value=None,
    )

    dy_interp = RegularGridInterpolator(
        (ys, xs),
        dy_grid,
        bounds_error=False,
        fill_value=None,
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


# -----------------------------
# Same movement calculation as IK_test.py
# but with separate pickup/place bias
# -----------------------------
def move_like_ik_test(
    servo_bus,
    x_in,
    y_in,
    z_target,
    correction_data,
    use_pickup_bias=False,
    extra_x_bias_mm=0.0,
    extra_y_bias_mm=0.0,
):
    dx_interp, dy_interp, xs, ys = correction_data

    target_x_mm = x_in * 25.4
    target_y_mm = y_in * 25.4

    corrected_x_mm, corrected_y_mm = apply_arm_correction(
        target_x_mm,
        target_y_mm,
        dx_interp,
        dy_interp,
        xs,
        ys,
    )

    # Pickup-only grasp bias.
    # This should NOT affect placement.
    if use_pickup_bias and target_x_mm < 0:
        corrected_x_mm += LEFT_PICKUP_GRASP_BIAS_MM
        print(
            f"Applied left-side PICKUP grasp bias: "
            f"{LEFT_PICKUP_GRASP_BIAS_MM:+.2f} mm"
        )

    # Optional placement/fine-tuning bias.
    # This is separate from the pickup bias.
    corrected_x_mm += extra_x_bias_mm
    corrected_y_mm += extra_y_bias_mm

    if extra_x_bias_mm != 0.0 or extra_y_bias_mm != 0.0:
        print(
            f"Applied extra bias: "
            f"dx={extra_x_bias_mm:+.2f} mm, dy={extra_y_bias_mm:+.2f} mm"
        )

    robot_x = corrected_x_mm
    robot_y = corrected_y_mm + base_offset

    IK_result = inverse_kinematics(
        robot_x,
        robot_y,
        L1,
        L2,
        L3,
        Z_offset,
        z_target,
    )

    if IK_result is None:
        print("The target position is unreachable.")
        return False

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

    servo_bus.move_time_write(1, base_cmd, 3.0)
    servo_bus.move_time_write(2, shoulder_cmd, 3.0)
    servo_bus.move_time_write(3, elbow_cmd, 3.0)
    servo_bus.move_time_write(4, wrist_cmd, 1.5)

    time.sleep(3.5)

    return True


# -----------------------------
# Gripper / rest helpers
# -----------------------------
def open_gripper(servo_bus):
    open_gripper_angle = cal_data["gripper"]["min_angle"]
    servo_bus.move_time_write(6, open_gripper_angle, 1.5)
    time.sleep(2)


def close_gripper(servo_bus):
    close_gripper_angle = cal_data["gripper"]["max_angle"]
    servo_bus.move_time_write(6, close_gripper_angle, 1.5)
    time.sleep(2)


def rest_arm(servo_bus, holding_block=False):
    print("Fold to resting position...")

    for servo_id in [1, 2, 3, 4, 5]:
        name = servo_names[servo_id]
        state = RESTING_STATES[servo_id]
        rest_angle = cal_data[name][f"{state}_angle"]
        servo_bus.move_time_write(servo_id, rest_angle, 2.0)

    time.sleep(2.5)

    if holding_block:
        print("Keeping claw closed because it is holding a block.")
    else:
        print("Closing claw at rest...")
        close_gripper(servo_bus)

    print("Rest complete.")


# -----------------------------
# Input helpers
# -----------------------------
def ask_float(prompt):
    while True:
        value = input(prompt).strip().lower()

        if value in ["q", "quit"]:
            return "quit"

        try:
            return float(value)
        except ValueError:
            print("Type a number or q to quit.")


# -----------------------------
# Main
# -----------------------------
def main():
    correction_data = load_arm_correction_interpolator()

    print("\nSimple Tower Stack")
    print("------------------")
    print("This moves like IK_test.py.")
    print(f"Pickup uses Z_target = {Z_target} mm.")
    print(f"Each new block automatically increases placement Z by {BLOCK_HEIGHT_MM} mm.")
    print("Pickup uses left-side grasp bias only when grabbing.")
    print("Placement does NOT use pickup grasp bias.")
    print("First block is stack level 0.")
    print("Type q to quit.\n")

    input("Press ENTER to connect...")

    with ServoBus(MAC_PORT, baudrate=1000000, discard_echo=False) as servo_bus:
        # Start folded with claw closed.
        rest_arm(servo_bus, holding_block=False)

        stack_level = 0

        while True:
            print("\n==============================")
            print(f"Current stack level: {stack_level}")
            print(f"Current placement Z: {Z_target + stack_level * BLOCK_HEIGHT_MM:.2f} mm")
            print("==============================")

            # -----------------------------
            # Pick block
            # -----------------------------
            pick_x = ask_float("\nPick x in inches: ")
            if pick_x == "quit":
                break

            pick_y = ask_float("Pick y in inches: ")
            if pick_y == "quit":
                break

            input("\nPress ENTER to move to pickup...")

            print("Opening gripper...")
            open_gripper(servo_bus)

            success = move_like_ik_test(
                servo_bus,
                pick_x,
                pick_y,
                Z_target,
                correction_data,
                use_pickup_bias=True,
            )

            if not success:
                rest_arm(servo_bus, holding_block=False)
                continue

            print("Closing gripper to grab block...")
            close_gripper(servo_bus)

            print("Returning to rest while holding block...")
            rest_arm(servo_bus, holding_block=True)

            print("\nBlock should now be held.")
            print("Now enter where to place it.")

            # -----------------------------
            # Place block
            # -----------------------------
            place_x = ask_float("\nPlace x in inches: ")
            if place_x == "quit":
                break

            place_y = ask_float("Place y in inches: ")
            if place_y == "quit":
                break

            place_z = Z_target + stack_level * BLOCK_HEIGHT_MM

            print(f"\nThis block will be placed at stack level {stack_level}.")
            print(f"Placement Z: {place_z:.2f} mm")
            input("Press ENTER to place block...")

            success = move_like_ik_test(
                servo_bus,
                place_x,
                place_y,
                place_z,
                correction_data,
                use_pickup_bias=False,
                extra_x_bias_mm=PLACE_X_BIAS_MM,
                extra_y_bias_mm=PLACE_Y_BIAS_MM,
            )

            if not success:
                rest_arm(servo_bus, holding_block=True)
                continue

            print("Opening gripper to release block...")
            open_gripper(servo_bus)

            print("Returning to rest and closing claw...")
            rest_arm(servo_bus, holding_block=False)

            # Increase stack level only after successful placement.
            stack_level += 1
            print(f"\nNext block will be stack level {stack_level}.")

            again = input("\nStack another block? y/n: ").strip().lower()
            if again not in ["y", "yes"]:
                break

        print("\nFinal rest...")
        rest_arm(servo_bus, holding_block=False)

    print("\nTower stack script finished.")


if __name__ == "__main__":
    main()