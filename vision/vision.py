import cv2
import numpy as np
import cv2.aruco as aruco

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
cube_height_mm = 45.0    # measured cube height
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

mask = cv2.inRange(hsv, (0, 0, 50), (180, 255, 255))
blur = cv2.GaussianBlur(mask, (5, 5), 0)
M = cv2.moments(blur)
if M.get('m00', 0) != 0:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
else:
    cX, cY = 0, 0

# Signed coordinates relative to top-center origin with inverted x-axis
origin_x = analysis_img.shape[1] // 2
origin_y = 0
relative_x = origin_x - cX
relative_y = cY - origin_y

# Convert pixel coordinates to millimeters using ArUco tag ID 3 top-left to top-right distance
pixel_scale_mm = None
if marker_3_corners is not None:
    # choose top-left and top-right based on x-coordinate order
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
    relative_x_mm = (relative_x * pixel_scale_mm)
    relative_y_mm = (relative_y * pixel_scale_mm) - 45
    print(f"Signed COM relative to bottom-center in mm: (x={relative_x_mm:.2f}, y={relative_y_mm:.2f})")
    relative_x_in = relative_x_mm / 25.4
    relative_y_in = relative_y_mm / 25.4
    print(f"Signed COM relative to bottom-center in inches: (x={relative_x_in:.2f}, y={relative_y_in:.2f})")
else:
    relative_x_mm = None
    relative_y_mm = None

COM = cv2.circle(analysis_img.copy(), (cX, cY), 5, (0, 0, 255), 2)
coord_img = analysis_img.copy()
axis_color = (0, 255, 255)
origin_pt = (origin_x, origin_y)
cv2.line(coord_img, (origin_x, 0), (origin_x, coord_img.shape[0] - 1), axis_color, 2)
cv2.line(coord_img, (0, origin_y), (coord_img.shape[1] - 1, origin_y), axis_color, 2)
cv2.circle(coord_img, origin_pt, 5, (0, 255, 0), -1)
cv2.circle(coord_img, (cX, cY), 6, (0, 0, 255), -1)
cv2.line(coord_img, origin_pt, (cX, cY), (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(coord_img, f'Origin (top-center)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 2)
cv2.putText(coord_img, f'COM: ({cX}, {cY})', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(coord_img, f'x={relative_x}, y={relative_y} (x-left positive)', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
