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

corners, ids, rejected = detector.detectMarkers(cropped)
orig_corners, orig_ids = corners, ids
orig_display = cropped.copy()
source_points = [None, None, None, None]

corner_modes = {
    0: 'top-right',
    1: 'top-left',
    2: 'bottom-left',
    3: 'bottom-right',
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

    for idx, marker_id in enumerate(orig_ids.flatten()):
        marker_id = int(marker_id)
        if marker_id in corner_modes:
            mode = corner_modes[marker_id]
            marker_corners = orig_corners[idx].reshape((4, 2))
            selected_index = choose_corner_by_position(marker_corners, mode)
            pt = tuple(marker_corners[selected_index].astype(int))
            source_points[marker_id] = pt
            cv2.circle(orig_display, pt, 6, (0, 255, 0), -1)
            cv2.putText(orig_display, f"ID{marker_id}", (pt[0] + 5, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Selected source points: {source_points}")
    cv2.imwrite('aruco_markers.jpg', orig_display)

analysis_img = cropped
analysis_name = 'cropped'

if None not in source_points:
    src_pts = np.array(source_points, dtype=np.float32)
    dst_pts = np.array([
        [640, 0],
        [0, 0],
        [0, 400],
        [640, 400]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(cropped, H, (640, 400))
    cv2.imwrite('warped.jpg', warped)
    analysis_img = warped
    analysis_name = 'warped'
else:
    print('Not enough valid source points for homography; skipping warp.')

hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, (85, 50, 50), (105, 255, 255))
blur = cv2.GaussianBlur(mask, (5, 5), 0)
M = cv2.moments(blur)
if M.get('m00', 0) != 0:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
else:
    cX, cY = 0, 0
COM = cv2.circle(analysis_img.copy(), (cX, cY), 5, (0, 0, 255), 2)
Middle = cv2.circle(cropped.copy(), (cropped.shape[1] // 2, cropped.shape[0] // 2), 5, (255, 0, 0), 2)
Bottom_left = cv2.circle(cropped.copy(), (40, cropped.shape[0]-40), 5, (0, 255, 0), 2)
Top_left = cv2.circle(cropped.copy(), (40, 40), 5, (255, 255, 0), 2)
Bottom_right = cv2.circle(cropped.copy(), (cropped.shape[1]-40, cropped.shape[0]-40), 5, (255, 0, 255), 2)
Top_right = cv2.circle(cropped.copy(), (cropped.shape[1]-40, 40), 5, (0, 255, 255), 2)
top_row = cv2.hconcat([Top_left, Top_right])
bottom_row = cv2.hconcat([Bottom_left, Bottom_right])
Corners_combined = cv2.vconcat([top_row, bottom_row])

print(f"Center: ({cX}, {cY})")
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
cap.release()
