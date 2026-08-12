"""
Color-to-Marker Matching (sorting task)

Objective: place 2-6 colored cubes anywhere on the mat, alongside their
matching ArUco "target" marker. The arm scans once, figures out which
colors have BOTH a cube and a target marker visible, and picks each cube
up and places it directly on top of its matching marker. No stacking --
every placement uses the same table-height Z.

VISION LOGIC in this file (camera capture, ArUco/board detection, color
masking, pixel-to-mm conversion) is copied over unchanged from Paul's
tested "new and improved" vision version -- do not modify those functions
without re-confirming they still work on real hardware.

ARM MOVEMENT LOGIC is rebuilt to match tests/final_rainbow_stack.py
exactly: pick a block (open gripper -> move to it -> close gripper ->
rest, holding the block), then place it on its marker (move -> open
gripper -> rest, empty claw), then repeat for the next color. Same
calibration loading, same servo names/resting states, same gripper
angles pulled from calibration_results.json, same `with ServoBus(...) as
servo_bus:` structure -- just with Z fixed at one level instead of
incrementing like the stack does.
"""

import sys
import time
import math
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
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--auto", action="store_true")
parser.add_argument("--color", type=str, default=None)
args = parser.parse_args()


# ============================================================
# Path setup -- same direct-import style as final_rainbow_stack.py.
# (The previous version of this file silently fell back to placeholder
# IK/servo functions when these imports failed, which is what let the
# "map_angle_to_servo(..., None)" bug ship without erroring loudly.)
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

L1 = 115.48
L2 = 135.81
L3 = 185.71
Z_offset = 127
PICK_Z_TARGET = 70   # height used when descending onto a cube to grab it
PLACE_Z_TARGET = 85  # height used when releasing over a marker -- higher
                       # than pickup so the block drops in cleanly instead
                       # of the claw needing to weave down near other
                       # cubes/markers already on the mat

base_offset = 23.11
base_angle_offset = 9

LEFT_PICKUP_GRASP_BIAS_MM = -9.5

# Placement bias is now side-dependent. The RIGHT-side values below are
# tuned and working well. The LEFT-side values are a starting point (same
# as the right side) -- tune these separately the same way
# LEFT_PICKUP_GRASP_BIAS_MM was tuned, by placing left-side markers and
# adjusting until the drop lands on-target.
PLACE_X_BIAS_MM = 59.32       # right side (x_mm >= 0)
PLACE_Y_BIAS_MM = -49.02      # right side (x_mm >= 0)

LEFT_PLACE_X_BIAS_MM = 12.32  # left side (x_mm < 0) -- NEEDS ITS OWN TUNING
LEFT_PLACE_Y_BIAS_MM = 4.02  # left side (x_mm < 0) -- NEEDS ITS OWN TUNING

# Safe resting movement (matches final_rainbow_stack.py exactly).
# After placing a block, rest_arm() rotates the BASE servo away by this
# many degrees FIRST, then lifts/folds the other joints, then swings the
# base back to its resting center -- so the claw clears the block
# vertically before the base does its big swing back, instead of
# dragging through it.
# If this moves the claw TOWARD the block/marker instead of away, flip
# this to a negative value.
SAFE_BASE_NUDGE_DEG = 5.0

SAFE_BASE_NUDGE_TIME = 1.0
SAFE_ARM_LIFT_TIME = 2.0
SAFE_BASE_REST_TIME = 1.5

# Tracks the last commanded base servo angle, so rest_arm() knows which
# direction "away" is. Set inside move_to_board_mm().
LAST_BASE_CMD = None

CROP_START_X = 0
CROP_START_Y = 0
CROP_END_X = 640
CROP_END_Y = 400

WARP_WIDTH = 640
WARP_HEIGHT = 400

TAG_3_TOP_EDGE_MM = 24.75
Y_COORD_OFFSET_MM = -45.0
X_SIGN = -1

MIN_MASK_AREA = 30
CORNER_MARKER_IDS = {0, 1, 2, 3}
MARKER_ID_TO_COLOR = {
    4: "red",
    5: "orange",
    6: "yellow",
    7: "green",
    8: "blue",
    9: "purple",
}


# ============================================================
# Servo names / rest states (same as final_rainbow_stack.py)
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
# Load calibration (same as final_rainbow_stack.py -- this was missing
# entirely in the previous version of this file, which is why the
# gripper had to be hardcoded to 0/180 instead of real calibrated angles)
# ============================================================
try:
    with open(CALIBRATION_DIR / "calibration_results.json", "r") as f:
        cal_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find calibration_results.json in {CALIBRATION_DIR}.")
    sys.exit(1)


# ============================================================
# Camera open and capture helpers -- VISION LOGIC, unchanged
# ============================================================
def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # Match camera defaults used in MultipleCOM.py for consistent images
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, -1)
    cap.set(cv2.CAP_PROP_CONTRAST, -1)
    cap.set(cv2.CAP_PROP_SATURATION, -1)
    if not cap.isOpened():
        print("Error: webcam not found.")
        sys.exit(1)
    return cap


def capture_cropped_frame(cap):
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: could not read frame from webcam.")
        return None, None
    cropped = frame[CROP_START_Y:CROP_END_Y, CROP_START_X:CROP_END_X]
    cv2.imwrite("Testimg.jpg", frame)
    cv2.imwrite("cropped_image.jpg", cropped)
    return frame, cropped


# ============================================================
# ArUco helpers -- VISION LOGIC, unchanged
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
        corners, ids, rejected = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
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
    samples = [arr.reshape(-1, 3) for arr in (top, bottom, left, right) if arr.size]
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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


# Vision: detect_and_warp_board and detect colored cubes using
# MultipleCOM logic -- VISION LOGIC, unchanged.
def detect_and_warp_board(cropped):
    corners, ids, rejected = detect_markers_compatible(cropped)
    if ids is None:
        print("No ArUco tags detected at all.")
        return None
    order = np.argsort(ids.flatten())
    ids = ids[order]
    corners = [corners[i] for i in order]
    display = aruco.drawDetectedMarkers(cropped.copy(), corners, ids)
    corner_modes = {0: 'bottom-left', 1: 'bottom-right', 2: 'top-right', 3: 'top-left'}
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
            mapping = {'top-right': 0, 'top-left': 1, 'bottom-left': 2, 'bottom-right': 3}
            source_points[mapping[mode]] = pt
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
            cv2.putText(display, f"ID{marker_id}", (pt[0] + 5, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite('aruco_markers.jpg', display)
    if None in source_points:
        print('Not all 4 corner markers detected; cannot build homography')
        return None
    src_pts = np.array(source_points, dtype=np.float32)
    dst_pts = np.array([[WARP_WIDTH, 0], [0, 0], [0, WARP_HEIGHT], [WARP_WIDTH, WARP_HEIGHT]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(cropped, H, (WARP_WIDTH, WARP_HEIGHT))
    cv2.imwrite('warped.jpg', warped)
    # re-detect in warped
    warped_corners, warped_ids, _ = detect_markers_compatible(warped)
    marker_pixel_centers = {}
    marker_3_corners_warped = None
    if warped_ids is not None:
        for idx, marker_id in enumerate(warped_ids.flatten()):
            marker_id = int(marker_id)
            marker_corners = np.array(warped_corners[idx], dtype=np.float32).reshape((4, 2))
            center = tuple(np.round(marker_corners.mean(axis=0)).astype(int))
            marker_pixel_centers[marker_id] = (float(center[0]), float(center[1]))
            if marker_id == 3:
                marker_3_corners_warped = marker_corners
    # fallback: project raw centers
    for idx, marker_id in enumerate(ids.flatten()):
        marker_id = int(marker_id)
        if marker_id in marker_pixel_centers:
            continue
        marker_corners = np.array(corners[idx], dtype=np.float32).reshape((4, 2))
        center = marker_corners.mean(axis=0).reshape(1, 1, 2)
        projected = cv2.perspectiveTransform(center, H).reshape(2)
        marker_pixel_centers[marker_id] = (float(projected[0]), float(projected[1]))
    # NOTE: marker_3_corners is intentionally left as the RAW (un-warped)
    # detection here, matching final_rainbow_stack.py's scale convention.
    # Warping stretches the image non-uniformly, so measuring tag 3's edge
    # length in the warped image gives a different (and here, much less
    # accurate) mm-per-pixel scale than measuring it in the raw camera
    # view -- and every downstream pick/place distance is multiplied by
    # this one number, plus arm_corrections.json / PLACE_X_BIAS_MM /
    # PLACE_Y_BIAS_MM were all tuned against the raw-view scale. Do not
    # overwrite marker_3_corners with marker_3_corners_warped or a
    # projected version here.
    cleaned = warped.copy()
    tag_mask_boxes_warped_space = []
    for idx, marker in enumerate(corners):
        warped_pts = cv2.perspectiveTransform(np.array(marker, dtype=np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
        tag_mask_boxes_warped_space.append(warped_pts)
    remove_claw(cleaned)
    cv2.imwrite('warped_cleaned.jpg', cleaned)
    tag_mask = np.zeros((WARP_HEIGHT, WARP_WIDTH), dtype=np.uint8)
    for warped_pts in tag_mask_boxes_warped_space:
        cv2.fillPoly(tag_mask, [warped_pts.astype(np.int32)], 255)
    return cleaned, marker_3_corners, marker_pixel_centers, tag_mask


# Color detection -- VISION LOGIC, unchanged.
rainbow_color_ranges = {
    "red": [((0, 40, 50), (5, 170, 255)), ((160, 40, 50), (180, 170, 255))],
    "orange": [((10, 60, 100), (25, 255, 255))],
    "yellow": [((80, 0, 230), (100, 40, 255))],
    "green": [((70, 80, 50), (99, 255, 255))],
    "blue": [((100, 50, 100), (120, 255, 255))],
    "purple": [((121, 70, 50), (159, 255, 255))],
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
        mask |= cv2.inRange(hsv_img, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    return mask


def detect_colored_cubes(analysis_img, origin_x, origin_y, pixel_scale_mm, tag_mask=None, marker_pixel_centers=None):
    hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
    cv2.imwrite('hsv.jpg', hsv)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    coord_img = analysis_img.copy()
    detections = {}

    # Draw coordinate axes like MultipleCOM's Center_of_Mass view.
    axis_color = (0, 255, 255)
    cv2.line(coord_img, (origin_x, 0), (origin_x, coord_img.shape[0] - 1), axis_color, 2)
    cv2.line(coord_img, (0, origin_y), (coord_img.shape[1] - 1, origin_y), axis_color, 2)
    cv2.circle(coord_img, (origin_x, origin_y), 5, (0, 255, 0), -1)

    for color_name, ranges in rainbow_color_ranges.items():
        mask = build_color_mask(hsv, ranges)
        if tag_mask is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(tag_mask))
            cv2.imwrite('tag_mask.jpg', tag_mask)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        blur = cv2.GaussianBlur(mask_clean, (5, 5), 0)
        cv2.imwrite(f'mask_{color_name}.jpg', mask_clean)
        M = cv2.moments(blur)
        if M.get('m00', 0) != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            detections[color_name] = {
                'pixel': (cX, cY),
                'relative_x': origin_x - cX,
                'relative_y': cY - origin_y,
                'mask': mask_clean,
            }
            cv2.circle(coord_img, (cX, cY), 6, color_draw_bgr[color_name], -1)
            cv2.putText(coord_img, f'{color_name}', (cX + 8, cY - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_draw_bgr[color_name], 2)
        else:
            detections[color_name] = None

    # Draw detected marker COMs on the same coordinate plane
    if marker_pixel_centers is not None:
        for marker_id, (px, py) in marker_pixel_centers.items():
            color_name = MARKER_ID_TO_COLOR.get(marker_id, 'white')
            draw_color = color_draw_bgr.get(color_name, (255, 255, 255))
            center = (int(round(px)), int(round(py)))
            cv2.circle(coord_img, center, 8, draw_color, 2)
            cv2.putText(coord_img, f'ID{marker_id}', (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

    cv2.imwrite('mask_combined.jpg', combined_mask)
    cv2.imwrite('color_coms.jpg', coord_img)
    cv2.imwrite('Center_of_Mass.jpg', coord_img)

    if pixel_scale_mm is not None:
        for color_name, data in detections.items():
            if data is None:
                continue
            data['relative_x_mm'] = data['relative_x'] * pixel_scale_mm
            data['relative_y_mm'] = data['relative_y'] * pixel_scale_mm + Y_COORD_OFFSET_MM
            data['relative_x_in'] = data['relative_x_mm'] / 25.4
            data['relative_y_in'] = data['relative_y_mm'] / 25.4

    return detections


def compute_pixel_scale_from_marker_3(marker_3_corners):
    if marker_3_corners is None:
        return None
    sorted_indices = np.argsort(marker_3_corners[:, 0])
    left_idx, right_idx = sorted_indices[0], sorted_indices[-1]
    top_pair = sorted([left_idx, right_idx], key=lambda i: marker_3_corners[i, 1])
    tl = marker_3_corners[top_pair[0]]
    tr = marker_3_corners[top_pair[1]]
    dist = np.linalg.norm(tr - tl)
    if dist <= 0:
        return None
    return TAG_3_TOP_EDGE_MM / dist


def pixel_to_mm(px, py, origin_x, origin_y, pixel_scale_mm):
    relative_x_px = X_SIGN * (px - origin_x)
    relative_y_px = py - origin_y
    x_mm = relative_x_px * pixel_scale_mm
    y_mm = relative_y_px * pixel_scale_mm + Y_COORD_OFFSET_MM
    return x_mm, y_mm


def compute_marker_targets(marker_pixel_centers, origin_x, origin_y, pixel_scale_mm):
    targets = {}
    for marker_id, (px, py) in marker_pixel_centers.items():
        x_mm, y_mm = pixel_to_mm(px, py, origin_x, origin_y, pixel_scale_mm)
        targets[marker_id] = {'marker_id': marker_id, 'relative_x_mm': x_mm, 'relative_y_mm': y_mm, 'pixel': (int(px), int(py))}
    return targets


# ============================================================
# Arm correction map (same as final_rainbow_stack.py, plus support for
# older correction files that used a "delta" key instead of "correction")
# ============================================================
def load_arm_correction_interpolator(path=CORRECTIONS_PATH):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found. Arm corrections disabled.")
        return (None, None, [], [])
    xs = sorted(set(float(item['target'][0]) for item in data))
    ys = sorted(set(float(item['target'][1]) for item in data))
    dx_grid = np.zeros((len(ys), len(xs)))
    dy_grid = np.zeros((len(ys), len(xs)))
    for item in data:
        tx = float(item['target'][0])
        ty = float(item['target'][1])
        ix = xs.index(tx)
        iy = ys.index(ty)
        if 'correction' in item:
            corr = item['correction']
        elif 'delta' in item:
            corr = item['delta']
        else:
            corr = [0.0, 0.0]
        dx_grid[iy, ix] = float(corr[0])
        dy_grid[iy, ix] = float(corr[1])
    dx_interp = RegularGridInterpolator((ys, xs), dx_grid, bounds_error=False, fill_value=None)
    dy_interp = RegularGridInterpolator((ys, xs), dy_grid, bounds_error=False, fill_value=None)
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
# Arm movement helpers -- rebuilt to match final_rainbow_stack.py
# exactly (same function signatures, same use of real cal_data, same
# print statements).
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
                f"Nudging base away from marker: "
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
        PICK_Z_TARGET,
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


def place_block_on_marker(servo_bus, color, target_data, correction_data):
    x_mm = target_data["relative_x_mm"]
    y_mm = target_data["relative_y_mm"]

    print("\n==============================")
    print(f"Placing {color.upper()} cube on its marker (ID {target_data['marker_id']})")
    print(f"Target: x={x_mm:.2f} mm, y={y_mm:.2f} mm")
    print("==============================")

    wait_if_needed(f"Press ENTER to place {color}...")

    # Placement bias is side-dependent: the arm's accuracy differs on the
    # left vs. right side of the workspace, same reasoning as
    # LEFT_PICKUP_GRASP_BIAS_MM for pickup. x_mm < 0 is the left side
    # (per X_SIGN = -1 above).
    if x_mm < 0:
        place_x_bias = LEFT_PLACE_X_BIAS_MM
        place_y_bias = LEFT_PLACE_Y_BIAS_MM
        print(f"Using LEFT-side placement bias: dx={place_x_bias:+.2f} mm, dy={place_y_bias:+.2f} mm")
    else:
        place_x_bias = PLACE_X_BIAS_MM
        place_y_bias = PLACE_Y_BIAS_MM
        print(f"Using RIGHT-side placement bias: dx={place_x_bias:+.2f} mm, dy={place_y_bias:+.2f} mm")

    success = move_to_board_mm(
        servo_bus,
        x_mm,
        y_mm,
        PLACE_Z_TARGET,
        correction_data,
        use_pickup_bias=False,
        extra_x_bias_mm=place_x_bias,
        extra_y_bias_mm=place_y_bias,
    )

    if not success:
        return False

    print("Opening gripper to release block...")
    open_gripper(servo_bus)

    print("Returning to rest with placement nudge and closing claw...")
    rest_arm(servo_bus, holding_block=False, safe_nudge=True)

    return True


# ============================================================
# Scan wrapper -- calls the (unchanged) vision functions above and
# fails the same way final_rainbow_stack.py's scan_cubes() does: no
# board, no scale, no arm movement.
# ============================================================
def scan():
    cap = open_camera()

    try:
        frame, cropped = capture_cropped_frame(cap)

        if frame is None:
            return None, None

        result = detect_and_warp_board(cropped)

        if result is None:
            print("Error: could not detect the board. Cannot scan.")
            return None, None

        cleaned, marker_3_corners, marker_pixel_centers, tag_mask = result

        pixel_scale_mm = compute_pixel_scale_from_marker_3(marker_3_corners)

        if pixel_scale_mm is None:
            print("Error: no pixel scale available. Cannot convert targets to mm.")
            return None, None

        origin_x = cleaned.shape[1] // 2
        origin_y = 0

        cube_detections = detect_colored_cubes(
            cleaned,
            origin_x,
            origin_y,
            pixel_scale_mm,
            tag_mask=tag_mask,
            marker_pixel_centers=marker_pixel_centers,
        )

        marker_targets = compute_marker_targets(
            marker_pixel_centers,
            origin_x,
            origin_y,
            pixel_scale_mm,
        )

        return cube_detections, marker_targets

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# Main -- same flow/shape as final_rainbow_stack.py's main(): scan,
# report, confirm, then pick->rest->place->rest per color inside one
# `with ServoBus(...) as servo_bus:` block.
# ============================================================
def main():
    print("\nColor-to-Marker Matching")
    print("-------------------------")
    print("Scanning for cubes and target markers (IDs 4-9)...")
    print(f"Port: {PORT}\n")

    correction_data = load_arm_correction_interpolator()

    cube_detections, marker_targets = scan()

    if cube_detections is None:
        print("Scan failed.")
        return

    # compute_marker_targets() (vision-side) keys by marker ID, not
    # color -- re-index by color here in the arm/orchestration code.
    marker_targets_by_color = {
        MARKER_ID_TO_COLOR[mid]: info
        for mid, info in marker_targets.items()
        if mid in MARKER_ID_TO_COLOR
    }

    all_colors = sorted(set(cube_detections) | set(marker_targets_by_color))

    matched_colors = []
    print("\n--- SCAN REPORT ---")
    for color in all_colors:
        has_cube = cube_detections.get(color) is not None
        has_marker = color in marker_targets_by_color

        if has_cube and has_marker:
            status = "MATCHED -- will pick and place"
            matched_colors.append(color)
        elif has_cube and not has_marker:
            status = "cube seen, no matching marker -- skipping"
        elif has_marker and not has_cube:
            status = "marker seen, no matching cube -- skipping"
        else:
            status = "neither seen"

        print(f"{color.title():<8}: {status}")
    print("-------------------\n")

    if not matched_colors:
        print("No matched color pairs found. Nothing to do.")
        return

    # Closest-cube-first: grab whichever matched cube is nearest to the
    # arm's origin, then the next closest, and so on. Distance is measured
    # from the cube's pickup position only (not the marker/place target),
    # since that's what determines how far the claw has to reach for the
    # NEXT pick.
    def cube_distance_mm(color):
        cube_data = cube_detections[color]
        return math.hypot(cube_data["relative_x_mm"], cube_data["relative_y_mm"])

    matched_colors.sort(key=cube_distance_mm)

    print("--- PICK ORDER (closest cube first) ---")
    for color in matched_colors:
        print(f"{color.title():<8}: {cube_distance_mm(color):.2f} mm from origin")
    print("----------------------------------------\n")

    print(f"Will process: {', '.join(c.title() for c in matched_colors)}\n")

    if args.dry_run:
        print("Dry-run mode enabled. IK will run, but servos will not move.")

    if not args.auto:
        input("Press ENTER to begin...")

    with ServoBus(PORT, baudrate=1000000, discard_echo=False) as servo_bus:
        rest_arm(servo_bus, holding_block=False)

        for color in matched_colors:
            cube_data = cube_detections[color]
            target_data = marker_targets_by_color[color]

            picked = pick_block(servo_bus, color, cube_data, correction_data)

            if not picked:
                print(f"Failed to pick {color}. Stopping.")
                rest_arm(servo_bus, holding_block=False)
                return

            placed = place_block_on_marker(servo_bus, color, target_data, correction_data)

            if not placed:
                print(f"Failed to place {color}. Stopping.")
                rest_arm(servo_bus, holding_block=True)
                return

            print(f"\nFinished {color}.\n")

        print("\nFinal rest...")
        rest_arm(servo_bus, holding_block=False)

    print("\nColor-to-marker matching complete.")


if __name__ == "__main__":
    main()
