import cv2
import numpy as np
import cv2.aruco as aruco
import sys
import time
import math
import json  
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import argparse

cap = cv2.VideoCapture(0)

# Explicitly reset all camera properties to defaults
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      # Enable auto exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -1)          # Reset to default exposure
cap.set(cv2.CAP_PROP_GAIN, 0)               # Reset gain
cap.set(cv2.CAP_PROP_BRIGHTNESS, -1)        # Reset brightness
cap.set(cv2.CAP_PROP_CONTRAST, -1)          # Reset contrast
cap.set(cv2.CAP_PROP_SATURATION, -1)        # Reset saturation

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
board = aruco.GridBoard((4, 4), 0.04, 0.01, aruco_dict)
detector = aruco.ArucoDetector(aruco_dict, parameters)
source_points = []
marker_3_corners = None
if not cap.isOpened():
	print('Error')
else:
	print('Webcam Found')

ret, frame = cap.read()
print(frame.shape)
start_x, start_y = 0, 0
end_x, end_y = 640, 400
cropped_tags = frame[start_y:end_y, start_x:end_x]
cropped = frame[start_y:end_y, start_x:end_x]
print(cropped.shape)

camera_height_mm = 736.6 # measured camera height above the marker plane
cube_height_mm = 30.0    # measured cube height
camera_height_m = camera_height_mm / 1000.0
cube_com_height_m = (cube_height_mm / 1000.0) / 2.0
fx_est = cropped.shape[1]
fy_est = cropped.shape[1]
cx = cropped.shape[1] / 2.0
cy = cropped.shape[0] / 2.0
camera_matrix = np.array([[fx_est, 0, cx], [0, fy_est, cy], [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros((5, 1), dtype=np.float64)
board_rvec = None
board_tvec = None


def pixel_to_world_height(pixel, height_m, rvec, tvec):
    if rvec is None or tvec is None:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    x_norm = (pixel[0] - cx) / fx_est
    y_norm = (pixel[1] - cy) / fy_est
    ray_camera = np.array([x_norm, y_norm, 1.0], dtype=np.float64)

    RT = R.T
    denom = float(RT[2] @ ray_camera)
    if abs(denom) < 1e-6:
        return None

    num = height_m + float(RT[2] @ t)
    s = num / denom
    world_point = RT @ (s * ray_camera - t)
    return world_point

corners, ids, rejected = detector.detectMarkers(cropped)
orig_corners, orig_ids = corners, ids
orig_display = cropped.copy()
source_points = [None, None, None, None]

corner_indices = {
    'top-right': 0,
    'top-left': 1,
    'bottom-left': 2,
    'bottom-right': 3,
}

corner_modes = {
    0: 'bottom-left',
    1: 'bottom-right',
    2: 'top-right',
    3: 'top-left',
}

def choose_corner_by_position(marker_corners, mode):
    # marker_corners shape: (4,2)
    x = marker_corners[:, 0]
    y = marker_corners[:, 1]
    if mode == 'top-right':
        scores = x - y
    elif mode == 'top-left':
        scores = -x - y
    elif mode == 'bottom-left':
        scores = -x + y
    elif mode == 'bottom-right':
        scores = x + y
    else:
        scores = np.zeros(len(x))
    return int(np.argmax(scores))

if orig_ids is not None:
    # Sort markers by ID for stable ordering
    order = np.argsort(orig_ids.flatten())
    orig_ids = orig_ids[order]
    orig_corners = [orig_corners[i] for i in order]
    print(f"Detected {len(orig_ids)} tags: {orig_ids.flatten()}")
    orig_display = aruco.drawDetectedMarkers(cropped.copy(), orig_corners, orig_ids)

    marker_3_corners = None
    for idx, marker_id in enumerate(orig_ids.flatten()):
        marker_id = int(marker_id)
        if marker_id == 3:
            marker_3_corners = np.array(orig_corners[idx], dtype=np.float32).reshape((4, 2))
        if marker_id in corner_modes:
            mode = corner_modes[marker_id]
            marker_corners = orig_corners[idx].reshape((4, 2))
            selected_index = choose_corner_by_position(marker_corners, mode)
            pt = tuple(marker_corners[selected_index].astype(int))
            source_points[corner_indices[mode]] = pt
            cv2.circle(orig_display, pt, 6, (0, 255, 0), -1)
            cv2.putText(orig_display, f"ID{marker_id}", (pt[0] + 5, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Selected source points: {source_points}")
    cv2.imwrite('aruco_markers.jpg', orig_display)

    if len(orig_ids) > 0:
        retval, board_rvec, board_tvec = aruco.estimatePoseBoard(orig_corners, orig_ids, board, camera_matrix, dist_coeffs, None, None)
        if retval:
            print(f"Board pose estimated: rvec={board_rvec.ravel()}, tvec={board_tvec.ravel()}")
        else:
            print("Board pose estimation failed")

analysis_img = cropped
analysis_name = 'cropped'

if None not in source_points:
    def perimeter_color(img, x1, y1, x2, y2, pad=12):
        h, w = img.shape[:2]
        top = img[max(0, y1-pad):y1, x1:x2]
        bottom = img[y2:min(h, y2+pad), x1:x2]
        left = img[y1:y2, max(0, x1-pad):x1]
        right = img[y1:y2, x2:min(w, x2+pad)]
        samples = [arr.reshape(-1, 3) for arr in (top, bottom, left, right) if arr.size]
        if not samples:
            return np.array([0, 0, 0], dtype=np.uint8)
        pixels = np.vstack(samples)
        return np.mean(pixels, axis=0).astype(np.uint8)

    src_pts = np.array(source_points, dtype=np.float32)
    dst_pts = np.array([
        [640, 0],   # top-right
        [0, 0],     # top-left
        [0, 400],   # bottom-left
        [640, 400]  # bottom-right
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(cropped, H, (640, 400))
    cv2.imwrite('warped.jpg', warped)

    cleaned = warped.copy()
    for corners in orig_corners:
        pts = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.int32)
        x, y, w, h = cv2.boundingRect(warped_pts)
        padding = 80
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, warped.shape[1])
        y2 = min(y + h + padding, warped.shape[0])

        fill_color = perimeter_color(warped, x1, y1, x2, y2, pad=12)
        cleaned[y1:y2, x1:x2] = fill_color

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

    remove_claw(cleaned)
    cv2.imwrite('warped_cleaned.jpg', cleaned)

    analysis_img = cleaned
    analysis_name = 'warped_cleaned'
else:
    print('Not enough valid source points for homography; skipping warp.')

hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)

rainbow_color_ranges = {
    "red": [((0, 100, 50), (10, 255, 255)), ((160, 100, 50), (180, 255, 255))],
    "orange": [((11, 100, 50), (20, 255, 255))],
    "yellow": [((21, 100, 50), (30, 255, 255))],
    "green": [((31, 100, 50), (75, 255, 255))],
    "blue": [((96, 100, 50), (130, 255, 255))],
    "purple": [((131, 100, 50), (169, 255, 255))],
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


combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
color_coms = {}
coord_img = analysis_img.copy()
origin_x = analysis_img.shape[1] // 2
origin_y = 0
axis_color = (0, 255, 255)
cv2.line(coord_img, (origin_x, 0), (origin_x, coord_img.shape[0] - 1), axis_color, 2)
cv2.line(coord_img, (0, origin_y), (coord_img.shape[1] - 1, origin_y), axis_color, 2)
cv2.circle(coord_img, (origin_x, origin_y), 5, (0, 255, 0), -1)

for color_name, ranges in rainbow_color_ranges.items():
    mask = build_color_mask(hsv, ranges)
    combined_mask = cv2.bitwise_or(combined_mask, mask)
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    M = cv2.moments(blur)
    if M.get('m00', 0) != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        color_coms[color_name] = {
            'pixel': (cX, cY),
            'relative_x': origin_x - cX,
            'relative_y': cY - origin_y,
            'mask': mask,
        }
        cv2.circle(coord_img, (cX, cY), 6, color_draw_bgr[color_name], -1)
        cv2.putText(coord_img, f'{color_name}', (cX + 8, cY - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_draw_bgr[color_name], 2)
    else:
        color_coms[color_name] = None
    cv2.imwrite(f'mask_{color_name}.jpg', mask)

cv2.imwrite('mask_combined.jpg', combined_mask)

pixel_scale_mm = None
if marker_3_corners is not None:
    sorted_indices = np.argsort(marker_3_corners[:, 0])
    left_idx, right_idx = sorted_indices[0], sorted_indices[-1]
    top_pair = sorted([left_idx, right_idx], key=lambda i: marker_3_corners[i, 1])
    tl = marker_3_corners[top_pair[0]]
    tr = marker_3_corners[top_pair[1]]
    pixel_distance = np.linalg.norm(tr - tl)
    if pixel_distance > 0:
        pixel_scale_mm = 24.75 / pixel_distance
        print(f"Scale: {pixel_scale_mm:.4f} mm/pixel from tag 3 top edge")
    else:
        print("Warning: pixel distance for tag 3 top edge is zero")
else:
    print("Warning: tag ID 3 not detected; cannot compute mm scale")

if pixel_scale_mm is not None:
    for color_name, data in color_coms.items():
        if data is None:
            print(f"{color_name.title()}: no object detected")
            continue
        x_mm = data['relative_x'] * pixel_scale_mm
        y_mm = data['relative_y'] * pixel_scale_mm - 45
        x_in = x_mm / 25.4
        y_in = y_mm / 25.4
        data['relative_x_mm'] = x_mm
        data['relative_y_mm'] = y_mm
        data['relative_x_in'] = x_in
        data['relative_y_in'] = y_in
        print(f"{color_name.title()} COM: pixel=({data['pixel'][0]}, {data['pixel'][1]}), mm=(x={x_mm:.2f}, y={y_mm:.2f}), in=(x={x_in:.2f}, y={y_in:.2f})")
else:
    for color_name, data in color_coms.items():
        if data is not None:
            print(f"{color_name.title()} COM: pixel=({data['pixel'][0]}, {data['pixel'][1]})")

cv2.imwrite('color_coms.jpg', coord_img)

# choose a single color target for downstream arm code
selected_color = None
args_target_color = None
try:
    args_target_color = args.target_color.lower()
except NameError:
    args_target_color = None

if args_target_color is None:
    selected_color = 'red'
else:
    selected_color = args_target_color if args_target_color in color_coms else 'red'
    if args_target_color not in color_coms:
        print(f"Warning: target color '{args_target_color}' not recognized. Defaulting to red.")

if color_coms[selected_color] is None:
    available = [name for name, data in color_coms.items() if data is not None]
    if available:
        selected_color = available[0]
        print(f"Warning: no {args_target_color or 'selected'} COM detected. Using {selected_color} instead.")
    else:
        print("Error: no colored COMs detected. Cannot continue to arm targeting.")
        relative_x_mm = None
        relative_y_mm = None
        relative_x_in = None
        relative_y_in = None

if pixel_scale_mm is not None and color_coms.get(selected_color) is not None:
    relative_x = color_coms[selected_color]['relative_x']
    relative_y = color_coms[selected_color]['relative_y']
    relative_x_mm = color_coms[selected_color]['relative_x_mm']
    relative_y_mm = color_coms[selected_color]['relative_y_mm']
    relative_x_in = color_coms[selected_color]['relative_x_in']
    relative_y_in = color_coms[selected_color]['relative_y_in']
    print(f"Selected target color: {selected_color}")
    print(f"Signed COM relative to bottom-center in mm: (x={relative_x_mm:.2f}, y={relative_y_mm:.2f})")
    print(f"Signed COM relative to bottom-center in inches: (x={relative_x_in:.2f}, y={relative_y_in:.2f})")
else:
    relative_x = 0
    relative_y = 0

COM = cv2.circle(coord_img.copy(), (origin_x - relative_x, origin_y + relative_y), 5, (0, 0, 255), 2)
coord_img = coord_img.copy()
cv2.circle(coord_img, (origin_x - relative_x, origin_y + relative_y), 6, (0, 0, 255), -1)
cv2.line(coord_img, (origin_x, origin_y), (origin_x - relative_x, origin_y + relative_y), (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(coord_img, f'Selected: {selected_color}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 2)
cv2.putText(coord_img, f'COM: ({origin_x - relative_x}, {origin_y + relative_y})', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(coord_img, f'x={relative_x}, y={relative_y} (x-left positive)', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.imwrite('coord_img.jpg', coord_img)

Middle = cv2.circle(cropped.copy(), (cropped.shape[1] // 2, cropped.shape[0] // 2), 5, (255, 0, 0), 2)
Bottom_left = cv2.circle(cropped.copy(), (40, cropped.shape[0]-40), 5, (0, 255, 0), 2)
Top_left = cv2.circle(cropped.copy(), (40, 40), 5, (255, 255, 0), 2)
Bottom_right = cv2.circle(cropped.copy(), (cropped.shape[1]-40, cropped.shape[0]-40), 5, (255, 0, 255), 2)
Top_right = cv2.circle(cropped.copy(), (cropped.shape[1]-40, 40), 5, (0, 255, 255), 2)
top_row = cv2.hconcat([Top_left, Top_right])
bottom_row = cv2.hconcat([Bottom_left, Bottom_right])
Corners_combined = cv2.vconcat([top_row, bottom_row])

print(f"Center: ({cX}, {cY})")
print(f"Signed COM relative to bottom-center: (x={relative_x}, y={relative_y})")
cv2.imwrite('cropped_image.jpg', cropped)
cv2.imwrite('hsv.jpg', hsv)
cv2.imwrite('Testimg.jpg', frame)
cv2.imwrite('blur.jpg', blur)
cv2.imwrite('mask.jpg', mask)
cv2.imwrite('Center_of_Mass.jpg', COM)
cv2.imwrite('Middle.jpg', Middle)
cv2.imwrite('Bottom_left.jpg', Bottom_left)
cv2.imwrite('Top_left.jpg', Top_left)
cv2.imwrite('Bottom_right.jpg', Bottom_right)
cv2.imwrite('Top_right.jpg', Top_right)
cv2.imwrite('Corners.jpg', Corners_combined)
cv2.imwrite('coord_img.jpg', coord_img)
cap.release()
parser = argparse.ArgumentParser()
parser.add_argument('--x', type=float, default=0.0, help="Target X on the board")
parser.add_argument('--y', type=float, default=9, help="Target Y on the board")
args = parser.parse_args()


lib_path = Path(__file__).resolve().parents[1] / "core" 
sys.path.insert(0, str(lib_path))
from IK_solver import inverse_kinematics, map_angle_to_servo

lib_path = Path(__file__).resolve().parents[1] / "lib" / "lewansoul-servo-bus-master" / "src" / "python"
sys.path.insert(0, str(lib_path))

from lewansoul_servo_bus import ServoBus

PORT = '/dev/ttyACM0'  

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


CORRECTIONS_PATH = "arm_corrections.json"


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

    # Interpolators expect input as (y, x), because grid shape is [y][x]
    dx_interp = RegularGridInterpolator(
        (ys, xs),
        dx_grid,
        bounds_error=False,
        fill_value=None
    )

    dy_interp = RegularGridInterpolator(
        (ys, xs),
        dy_grid,
        bounds_error=False,
        fill_value=None
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

L1 = 115.48 # Length of the upper arm in mm
L2 = 135.81 # Length of the forearm in mm
L3 = 185.71 # Length of the end effector in mm
Z_offset = 127 # Offset in the Z direction in mm
Z_target = 70 # Target Z position in mm
base_offset = 23.11
base_angle_offset = 9

in_x = relative_x_in 
in_y = relative_y_in

target_x_mm = relative_x_mm 
target_y_mm = relative_y_mm

dx_interp, dy_interp, xs, ys = load_arm_correction_interpolator()

corrected_x_mm, corrected_y_mm = apply_arm_correction(
    target_x_mm,
    target_y_mm,
    dx_interp,
    dy_interp,
    xs,
    ys
)

LEFT_GRASP_BIAS_MM = -9.5

if target_x_mm < 0:
    corrected_x_mm += LEFT_GRASP_BIAS_MM
    print(f"Applied left-side grasp bias: {LEFT_GRASP_BIAS_MM:+.2f} mm")

robot_x = corrected_x_mm
robot_y = corrected_y_mm + base_offset


IK_result = inverse_kinematics(robot_x, robot_y, L1, L2, L3, Z_offset, Z_target)

if IK_result is None:
    print("The target position is unreachable. Please choose a different position.")
    sys.exit(1)

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

with ServoBus(PORT, baudrate=1000000, discard_echo=False) as servo_bus:

    # Move to target position
    open_gripper_angle = cal_data["gripper"]["min_angle"]
    servo_bus.move_time_write(6, open_gripper_angle, 1.5)
    time.sleep(2)
    
    servo_bus.move_time_write(1, base_cmd, 3.0)
    servo_bus.move_time_write(2, shoulder_cmd, 3.0)
    servo_bus.move_time_write(3, elbow_cmd, 3.0)
    servo_bus.move_time_write(4, wrist_cmd, 1.5)


    time.sleep(3.5)

    close_gripper_angle = cal_data["gripper"]["max_angle"]
    servo_bus.move_time_write(6, close_gripper_angle, 1.5)
    time.sleep(2)

    print("Fold to resting position...")
    for servo_id in [1, 2, 3, 4, 5]:
            name = servo_names[servo_id]
            state = RESTING_STATES[servo_id]
            
            rest_angle = cal_data[name][f"{state}_angle"]
            
            servo_bus.move_time_write(servo_id, rest_angle, 2.0)

        # Give the motors time to fold back up before the script ends and cuts communication
    time.sleep(2.5)
    print("Sequence complete.")
