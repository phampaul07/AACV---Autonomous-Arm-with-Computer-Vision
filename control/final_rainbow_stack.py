import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import cv2.aruco as aruco
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# Argument parser
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Run vision and IK, but do not move servos.",
)

parser.add_argument(
    "--scan-only",
    action="store_true",
    help="Only detect cube positions and print them. Do not move servos.",
)

parser.add_argument(
    "--auto",
    action="store_true",
    help="Run without asking before each pick/place.",
)

args = parser.parse_args()


# ============================================================
# Path setup
# ============================================================
core_path = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(core_path))
from IK_solver import inverse_kinematics, map_angle_to_servo

# calibration/ lives next to control/ at the repo root -- build the path
# from this script's own location instead of assuming a particular cwd.
CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"

servo_lib_path = (
    Path(__file__).resolve().parents[1]
    / "lib"
    / "lewansoul-servo-bus-master"
    / "src"
    / "python"
)
sys.path.insert(0, str(servo_lib_path))
from lewansoul_servo_bus import ServoBus


# ============================================================
# Main config
# ============================================================
# Fixed rather than CLI args -- this only ever runs on the Raspberry Pi
# against one webcam, so there's nothing to actually configure at runtime.
PORT = "/dev/ttyACM0"
CAMERA_INDEX = 0

CORRECTIONS_PATH = CALIBRATION_DIR / "arm_corrections.json"

# Same arm geometry as IK_test.py / tower_stack.py
L1 = 115.48
L2 = 135.81
L3 = 185.71

Z_offset = 127
Z_target = 70

base_offset = 23.11
base_angle_offset = 9

# Cube / stack settings
BLOCK_HEIGHT_MM = 30

BASE_COLOR = "purple"

# Bottom to top:
# purple, blue, green, yellow, orange, red
# Purple stays on the board as the base.
STACK_ORDER = ["blue", "green", "yellow", "orange", "red"]

# Pickup-only bias
LEFT_PICKUP_GRASP_BIAS_MM = -9.5

# Placement-only bias
PLACE_X_BIAS_MM = 44.8
PLACE_Y_BIAS_MM = -25.0

# Safe resting movement
# If this moves the claw TOWARD the tower, flip this to -12.0
SAFE_BASE_NUDGE_DEG = 5.0

SAFE_BASE_NUDGE_TIME = 1.0
SAFE_ARM_LIFT_TIME = 2.0
SAFE_BASE_REST_TIME = 1.5

# Red is the last/top cube. Place it a little higher so it clears the stack.
RED_EXTRA_Z_MM = 15.0

LAST_BASE_CMD = None

# Camera crop / warp
CROP_START_X = 0
CROP_START_Y = 0
CROP_END_X = 640
CROP_END_Y = 400

WARP_WIDTH = 640
WARP_HEIGHT = 400

TAG_3_TOP_EDGE_MM = 24.75

Y_COORD_OFFSET_MM = -45.0

# Your older code used pixel-left positive.
# If the arm moves mirrored left/right, change this to 1.
X_SIGN = -1

MIN_MASK_AREA = 200


# ============================================================
# Servo names / rest states
# ============================================================
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


# ============================================================
# Load calibration
# ============================================================
try:
    with open(CALIBRATION_DIR / "calibration_results.json", "r") as f:
        cal_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find calibration_results.json in {CALIBRATION_DIR}.")
    sys.exit(1)


# ============================================================
# Camera helpers
# ============================================================
def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, -1)
    cap.set(cv2.CAP_PROP_CONTRAST, -1)
    cap.set(cv2.CAP_PROP_SATURATION, -1)

    if not cap.isOpened():
        print("Error: webcam not found.")
        sys.exit(1)

    print("Webcam found.")
    return cap


def capture_cropped_frame(cap):
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: could not read frame from webcam.")
        sys.exit(1)

    cropped = frame[CROP_START_Y:CROP_END_Y, CROP_START_X:CROP_END_X]

    print(f"Full frame shape: {frame.shape}")
    print(f"Cropped frame shape: {cropped.shape}")

    cv2.imwrite("Testimg.jpg", frame)
    cv2.imwrite("cropped_image.jpg", cropped)

    return frame, cropped


# ============================================================
# ArUco helpers
# ============================================================
def get_aruco_detector():
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        return aruco_dict, parameters, detector

    return aruco_dict, parameters, None


def detect_markers_compatible(img):
    aruco_dict, parameters, detector = get_aruco_detector()

    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(img)
    else:
        corners, ids, rejected = aruco.detectMarkers(
            img,
            aruco_dict,
            parameters=parameters,
        )

    return corners, ids, rejected


def choose_corner_by_position(marker_corners, mode):
    x = marker_corners[:, 0]
    y = marker_corners[:, 1]

    if mode == "top-right":
        scores = x - y
    elif mode == "top-left":
        scores = -x - y
    elif mode == "bottom-left":
        scores = -x + y
    elif mode == "bottom-right":
        scores = x + y
    else:
        scores = np.zeros(len(x))

    return int(np.argmax(scores))


def perimeter_color(img, x1, y1, x2, y2, pad=12):
    h, w = img.shape[:2]

    top = img[max(0, y1 - pad):y1, x1:x2]
    bottom = img[y2:min(h, y2 + pad), x1:x2]
    left = img[y1:y2, max(0, x1 - pad):x1]
    right = img[y1:y2, x2:min(w, x2 + pad)]

    samples = [
        arr.reshape(-1, 3)
        for arr in (top, bottom, left, right)
        if arr.size
    ]

    if not samples:
        return np.array([0, 0, 0], dtype=np.uint8)

    pixels = np.vstack(samples)
    return np.mean(pixels, axis=0).astype(np.uint8)


def remove_claw(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]

    mask = np.zeros_like(thresh)

    top_limit = int(h * 0.35)
    mask[0:top_limit, :] = thresh[0:top_limit, :]

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return False

    claw_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(claw_contour) < 500:
        return False

    x, y, w_box, h_box = cv2.boundingRect(claw_contour)

    padding = 55
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w_box + padding, img.shape[1])
    y2 = min(y + h_box + padding, img.shape[0])

    fill_color = perimeter_color(img, x1, y1, x2, y2, pad=12)
    img[y1:y2, x1:x2] = fill_color

    return True


def detect_and_warp_board(cropped):
    corners, ids, rejected = detect_markers_compatible(cropped)

    if ids is None:
        print("No ArUco tags detected. Using cropped image without warp.")
        return cropped, None, None

    order = np.argsort(ids.flatten())
    ids = ids[order]
    corners = [corners[i] for i in order]

    print(f"Detected {len(ids)} tags: {ids.flatten()}")

    display = aruco.drawDetectedMarkers(cropped.copy(), corners, ids)

    corner_indices = {
        "top-right": 0,
        "top-left": 1,
        "bottom-left": 2,
        "bottom-right": 3,
    }

    corner_modes = {
        0: "bottom-left",
        1: "bottom-right",
        2: "top-right",
        3: "top-left",
    }

    source_points = [None, None, None, None]
    marker_3_corners = None

    for idx, marker_id in enumerate(ids.flatten()):
        marker_id = int(marker_id)
        marker_corners = np.array(corners[idx], dtype=np.float32).reshape((4, 2))

        if marker_id == 3:
            marker_3_corners = marker_corners

        if marker_id in corner_modes:
            mode = corner_modes[marker_id]
            selected_index = choose_corner_by_position(marker_corners, mode)
            pt = tuple(marker_corners[selected_index].astype(int))

            source_points[corner_indices[mode]] = pt

            cv2.circle(display, pt, 6, (0, 255, 0), -1)
            cv2.putText(
                display,
                f"ID{marker_id}",
                (pt[0] + 5, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    print(f"Selected source points: {source_points}")
    cv2.imwrite("aruco_markers.jpg", display)

    if None in source_points:
        print("Not enough valid source points for homography. Stopping.")
        sys.exit(1)

    src_pts = np.array(source_points, dtype=np.float32)

    dst_pts = np.array(
        [
            [WARP_WIDTH, 0],
            [0, 0],
            [0, WARP_HEIGHT],
            [WARP_WIDTH, WARP_HEIGHT],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(cropped, H, (WARP_WIDTH, WARP_HEIGHT))
    cv2.imwrite("warped.jpg", warped)

    cleaned = warped.copy()

    for marker in corners:
        pts = np.array(marker, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.int32)

        x, y, w, h = cv2.boundingRect(warped_pts)

        padding = 80
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, warped.shape[1])
        y2 = min(y + h + padding, warped.shape[0])

        fill_color = perimeter_color(warped, x1, y1, x2, y2, pad=12)
        cleaned[y1:y2, x1:x2] = fill_color

    remove_claw(cleaned)

    cv2.imwrite("warped_cleaned.jpg", cleaned)

    return cleaned, corners, marker_3_corners


# ============================================================
# Color detection
# ============================================================
rainbow_color_ranges = {
    "red": [
        ((0, 40, 50), (5, 255, 255)),
        ((160, 40, 50), (180, 255, 255)),
    ],
    "orange": [
        ((12, 60, 100), (30, 255, 255)),
    ],
    "yellow": [
        ((80, 0, 230), (100, 40, 255)),
    ],
    "green": [
        ((75, 80, 50), (89, 255, 255)),
    ],
    "blue": [
        ((90, 50, 100), (120, 255, 255)),
    ],
    "purple": [
        ((121, 70, 50), (159, 255, 255)),
    ],
}

color_draw_bgr = {
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "purple": (211, 0, 148),
}


def build_color_mask(hsv_img, ranges):
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    for low, high in ranges:
        mask |= cv2.inRange(
            hsv_img,
            np.array(low, dtype=np.uint8),
            np.array(high, dtype=np.uint8),
        )

    return mask


def compute_pixel_scale_from_marker_3(marker_3_corners):
    if marker_3_corners is None:
        print("Warning: tag ID 3 not detected. Cannot compute mm scale.")
        return None

    sorted_indices = np.argsort(marker_3_corners[:, 0])
    left_idx = sorted_indices[0]
    right_idx = sorted_indices[-1]

    top_pair = sorted(
        [left_idx, right_idx],
        key=lambda i: marker_3_corners[i, 1],
    )

    tl = marker_3_corners[top_pair[0]]
    tr = marker_3_corners[top_pair[1]]

    pixel_distance = np.linalg.norm(tr - tl)

    if pixel_distance <= 0:
        print("Warning: pixel distance for tag 3 top edge is zero.")
        return None

    pixel_scale_mm = TAG_3_TOP_EDGE_MM / pixel_distance
    print(f"Scale: {pixel_scale_mm:.4f} mm/pixel from tag 3 top edge")

    return pixel_scale_mm


def detect_colored_objects(analysis_img, marker_3_corners):
    hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv.jpg", hsv)

    pixel_scale_mm = compute_pixel_scale_from_marker_3(marker_3_corners)

    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    coord_img = analysis_img.copy()

    origin_x = analysis_img.shape[1] // 2
    origin_y = 0

    axis_color = (0, 255, 255)
    cv2.line(coord_img, (origin_x, 0), (origin_x, coord_img.shape[0] - 1), axis_color, 2)
    cv2.line(coord_img, (0, origin_y), (coord_img.shape[1] - 1, origin_y), axis_color, 2)
    cv2.circle(coord_img, (origin_x, origin_y), 5, (0, 255, 0), -1)

    detections = {}

    for color_name, ranges in rainbow_color_ranges.items():
        mask = build_color_mask(hsv, ranges)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        M = cv2.moments(blur)

        cv2.imwrite(f"mask_{color_name}.jpg", mask)

        if M.get("m00", 0) < MIN_MASK_AREA:
            detections[color_name] = None
            print(f"{color_name.title()}: no object detected")
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        relative_x_px = X_SIGN * (cX - origin_x)
        relative_y_px = cY - origin_y

        detection = {
            "pixel": (cX, cY),
            "relative_x_px": relative_x_px,
            "relative_y_px": relative_y_px,
        }

        if pixel_scale_mm is not None:
            x_mm = relative_x_px * pixel_scale_mm
            y_mm = relative_y_px * pixel_scale_mm + Y_COORD_OFFSET_MM

            detection["relative_x_mm"] = x_mm
            detection["relative_y_mm"] = y_mm
            detection["relative_x_in"] = x_mm / 25.4
            detection["relative_y_in"] = y_mm / 25.4

            print(
                f"{color_name.title()} COM: "
                f"pixel=({cX}, {cY}), "
                f"mm=(x={x_mm:.2f}, y={y_mm:.2f}), "
                f"in=(x={x_mm / 25.4:.2f}, y={y_mm / 25.4:.2f})"
            )
        else:
            print(f"{color_name.title()} COM: pixel=({cX}, {cY})")

        detections[color_name] = detection

        cv2.circle(coord_img, (cX, cY), 6, color_draw_bgr[color_name], -1)
        cv2.putText(
            coord_img,
            color_name,
            (cX + 8, cY - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_draw_bgr[color_name],
            2,
        )

    cv2.imwrite("mask_combined.jpg", combined_mask)
    cv2.imwrite("color_coms.jpg", coord_img)
    cv2.imwrite("coord_img.jpg", coord_img)

    return detections, pixel_scale_mm


def print_detected_coordinates(detections):
    print("\n--- DETECTED CUBE COORDINATES ---")

    for color in ["purple", "blue", "green", "yellow", "orange", "red"]:
        data = detections.get(color)

        if data is None:
            print(f"{color.title()}: NOT DETECTED")
            continue

        print(
            f"{color.title()}: "
            f"x={data['relative_x_in']:.2f} in, "
            f"y={data['relative_y_in']:.2f} in "
            f"({data['relative_x_mm']:.2f} mm, {data['relative_y_mm']:.2f} mm)"
        )

    print("---------------------------------\n")


def validate_required_colors(detections):
    required = [BASE_COLOR] + STACK_ORDER
    missing = []

    for color in required:
        if detections.get(color) is None:
            missing.append(color)

    if missing:
        print("Error: missing required colors:")
        for color in missing:
            print(f"  - {color}")
        return False

    return True


# ============================================================
# Arm correction map
# ============================================================
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


def apply_arm_correction(x_mm, y_mm, correction_data):
    dx_interp, dy_interp, xs, ys = correction_data

    if dx_interp is None or dy_interp is None:
        return x_mm, y_mm

    x_clamped = clamp(x_mm, min(xs), max(xs))
    y_clamped = clamp(y_mm, min(ys), max(ys))

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


# ============================================================
# Arm movement helpers
# ============================================================
def compute_servo_commands(
    target_x_mm,
    target_y_mm,
    z_target,
    correction_data,
    use_pickup_bias=False,
    extra_x_bias_mm=0.0,
    extra_y_bias_mm=0.0,
):
    corrected_x_mm, corrected_y_mm = apply_arm_correction(
        target_x_mm,
        target_y_mm,
        correction_data,
    )

    if use_pickup_bias and target_x_mm < 0:
        corrected_x_mm += LEFT_PICKUP_GRASP_BIAS_MM
        print(
            f"Applied left-side PICKUP grasp bias: "
            f"{LEFT_PICKUP_GRASP_BIAS_MM:+.2f} mm"
        )

    corrected_x_mm += extra_x_bias_mm
    corrected_y_mm += extra_y_bias_mm

    if extra_x_bias_mm != 0.0 or extra_y_bias_mm != 0.0:
        print(
            f"Applied placement bias: "
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
        return None

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

    return base_cmd, shoulder_cmd, elbow_cmd, wrist_cmd


def move_to_board_mm(
    servo_bus,
    x_mm,
    y_mm,
    z_target,
    correction_data,
    use_pickup_bias=False,
    extra_x_bias_mm=0.0,
    extra_y_bias_mm=0.0,
):
    global LAST_BASE_CMD

    commands = compute_servo_commands(
        x_mm,
        y_mm,
        z_target,
        correction_data,
        use_pickup_bias=use_pickup_bias,
        extra_x_bias_mm=extra_x_bias_mm,
        extra_y_bias_mm=extra_y_bias_mm,
    )

    if commands is None:
        return False

    base_cmd, shoulder_cmd, elbow_cmd, wrist_cmd = commands

    LAST_BASE_CMD = base_cmd

    if args.dry_run:
        print("Dry run enabled. Not moving servos.")
        return True

    servo_bus.move_time_write(1, base_cmd, 3.0)
    servo_bus.move_time_write(2, shoulder_cmd, 3.0)
    servo_bus.move_time_write(3, elbow_cmd, 3.0)
    servo_bus.move_time_write(4, wrist_cmd, 1.5)

    time.sleep(3.5)

    return True


def get_rest_angle(servo_id):
    name = servo_names[servo_id]
    state = RESTING_STATES[servo_id]
    return cal_data[name][f"{state}_angle"]


def open_gripper(servo_bus):
    if args.dry_run:
        print("Dry run: open gripper")
        return

    open_gripper_angle = cal_data["gripper"]["min_angle"]
    servo_bus.move_time_write(6, open_gripper_angle, 1.5)
    time.sleep(2)


def close_gripper(servo_bus):
    if args.dry_run:
        print("Dry run: close gripper")
        return

    close_gripper_angle = cal_data["gripper"]["max_angle"]
    servo_bus.move_time_write(6, close_gripper_angle, 1.5)
    time.sleep(2)


def rest_arm(servo_bus, holding_block=False, safe_nudge=False):
    global LAST_BASE_CMD

    print("Retreating to resting position...")

    if not args.dry_run:
        # Step 1: Optional base nudge.
        # Only turn this on after placement, not after pickup.
        if safe_nudge and LAST_BASE_CMD is not None:
            base_name = servo_names[1]
            base_min = cal_data[base_name].get("min_angle", -999)
            base_max = cal_data[base_name].get("max_angle", 999)

            safe_base_cmd = LAST_BASE_CMD + SAFE_BASE_NUDGE_DEG
            safe_base_cmd = clamp(safe_base_cmd, base_min, base_max)

            print(
                f"Nudging base away from tower: "
                f"{LAST_BASE_CMD:.2f}° -> {safe_base_cmd:.2f}°"
            )

            servo_bus.move_time_write(1, safe_base_cmd, SAFE_BASE_NUDGE_TIME)
            time.sleep(SAFE_BASE_NUDGE_TIME + 0.3)

            LAST_BASE_CMD = safe_base_cmd

        # Step 2: Lift/fold shoulder, elbow, wrist.
        print("Lifting/folding arm...")

        for servo_id in [2, 3, 4, 5]:
            rest_angle = get_rest_angle(servo_id)
            servo_bus.move_time_write(servo_id, rest_angle, SAFE_ARM_LIFT_TIME)

        time.sleep(SAFE_ARM_LIFT_TIME + 0.5)

        # Step 3: Return base to resting center after arm is lifted.
        print("Returning base to resting center...")

        base_rest_angle = get_rest_angle(1)
        servo_bus.move_time_write(1, base_rest_angle, SAFE_BASE_REST_TIME)

        time.sleep(SAFE_BASE_REST_TIME + 0.3)

        LAST_BASE_CMD = base_rest_angle

    # Step 4: Handle gripper.
    if holding_block:
        print("Keeping claw closed because it is holding a block.")
    else:
        print("Closing claw at rest...")
        close_gripper(servo_bus)

    print("Rest complete.")

def wait_if_needed(message):
    if args.auto or args.dry_run:
        return

    input(message)


def pick_block(servo_bus, color, cube_data, correction_data):
    x_mm = cube_data["relative_x_mm"]
    y_mm = cube_data["relative_y_mm"]

    print("\n==============================")
    print(f"Picking {color.upper()} cube")
    print(f"Pickup target: x={x_mm:.2f} mm, y={y_mm:.2f} mm")
    print("==============================")

    wait_if_needed(f"Press ENTER to pick {color}...")

    print("Opening gripper...")
    open_gripper(servo_bus)

    success = move_to_board_mm(
        servo_bus,
        x_mm,
        y_mm,
        Z_target,
        correction_data,
        use_pickup_bias=True,
    )

    if not success:
        return False

    print("Closing gripper to grab block...")
    close_gripper(servo_bus)

    print("Returning to rest while holding block...")
    rest_arm(servo_bus, holding_block=True, safe_nudge=False)

    return True


def place_block_on_stack(
    servo_bus,
    color,
    stack_x_mm,
    stack_y_mm,
    stack_level,
    correction_data,
):
    place_z = Z_target + stack_level * BLOCK_HEIGHT_MM

    if color == "red":
        place_z += RED_EXTRA_Z_MM
        print(f"Applied extra red placement height: +{RED_EXTRA_Z_MM:.2f} mm")

    print("\n==============================")
    print(f"Placing {color.upper()} cube on stack")
    print(f"Stack target: x={stack_x_mm:.2f} mm, y={stack_y_mm:.2f} mm")
    print(f"Stack level: {stack_level}")
    print(f"Placement Z: {place_z:.2f} mm")
    print("==============================")

    wait_if_needed(f"Press ENTER to place {color}...")

    success = move_to_board_mm(
        servo_bus,
        stack_x_mm,
        stack_y_mm,
        place_z,
        correction_data,
        use_pickup_bias=False,
        extra_x_bias_mm=PLACE_X_BIAS_MM,
        extra_y_bias_mm=PLACE_Y_BIAS_MM,
    )

    if not success:
        return False

    print("Opening gripper to release block...")
    open_gripper(servo_bus)

    print("Returning to rest with placement nudge and closing claw...")
    rest_arm(servo_bus, holding_block=False, safe_nudge=True)

    return True


# ============================================================
# Vision scan
# ============================================================
def scan_cubes():
    cap = open_camera()

    try:
        _, cropped = capture_cropped_frame(cap)
        analysis_img, _, marker_3_corners = detect_and_warp_board(cropped)

        detections, pixel_scale_mm = detect_colored_objects(
            analysis_img,
            marker_3_corners,
        )

        if pixel_scale_mm is None:
            print("Error: no pixel scale available. Cannot convert targets to mm.")
            return None

        print_detected_coordinates(detections)

        return detections

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================
def main():
    print("\nFinal Rainbow Stack with Safe Rest")
    print("----------------------------------")
    print("Goal stack order bottom to top:")
    print("purple, blue, green, yellow, orange, red")
    print("\nPurple is NOT picked up.")
    print("Purple COM becomes the stack/base coordinate.")
    print("Every other cube gets stacked onto purple.")
    print(f"Block height: {BLOCK_HEIGHT_MM} mm")
    print(f"Port: {PORT}")
    print(f"Safe base nudge: {SAFE_BASE_NUDGE_DEG:+.2f}°")
    print(f"Red extra placement height: +{RED_EXTRA_Z_MM:.2f} mm")
    print("")

    correction_data = load_arm_correction_interpolator()

    detections = scan_cubes()

    if detections is None:
        print("Scan failed.")
        return

    if not validate_required_colors(detections):
        print("Not moving arm because not all cubes were detected.")
        return

    base_data = detections[BASE_COLOR]

    stack_x_mm = base_data["relative_x_mm"]
    stack_y_mm = base_data["relative_y_mm"]

    print("\n--- STACK BASE ---")
    print(f"Base color: {BASE_COLOR}")
    print(f"Base coordinate: x={stack_x_mm:.2f} mm, y={stack_y_mm:.2f} mm")
    print(f"Base coordinate: x={stack_x_mm / 25.4:.2f} in, y={stack_y_mm / 25.4:.2f} in")
    print("------------------\n")

    if args.scan_only:
        print("Scan-only mode enabled. Not moving arm.")
        return

    if args.dry_run:
        print("Dry-run mode enabled. IK will run, but servos will not move.")

    if not args.auto:
        input("Press ENTER to begin full rainbow stack...")

    with ServoBus(PORT, baudrate=1000000, discard_echo=False) as servo_bus:
        rest_arm(servo_bus, holding_block=False, safe_nudge=False)

        # Purple is already level 0.
        # Blue goes on top of purple at level 1.
        stack_level = 1

        for color in STACK_ORDER:
            cube_data = detections[color]

            picked = pick_block(
                servo_bus,
                color,
                cube_data,
                correction_data,
            )

            if not picked:
                print(f"Failed to pick {color}. Stopping.")
                rest_arm(servo_bus, holding_block=False)
                return

            placed = place_block_on_stack(
                servo_bus,
                color,
                stack_x_mm,
                stack_y_mm,
                stack_level,
                correction_data,
            )

            if not placed:
                print(f"Failed to place {color}. Stopping.")
                rest_arm(servo_bus, holding_block=True)
                return

            stack_level += 1

        print(f"\nFinished {color}. Next stack level: {stack_level}\n")

    print("\nFinal safe rest...")
    rest_arm(servo_bus, holding_block=False)


if __name__ == "__main__":
    main()
