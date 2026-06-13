import cv2
import numpy as np
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print('Error')
else:
	print('Webcam Found')

ret, frame = cap.read()
print(frame.shape)
start_x, start_y = 120, 0
end_x, end_y = 520, 400


img = frame
cropped = frame[start_y:end_y, start_x:end_x]
print(cropped.shape)
rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
blur = cv2.GaussianBlur(mask, (5, 5), 0)
M = cv2.moments(blur)
if M.get('m00', 0) != 0:
	cX = int(M['m10'] / M['m00'])
	cY = int(M['m01'] / M['m00'])
else:
	cX, cY = 0, 0
COM = cv2.circle(cropped.copy(), (cX, cY), 5, (0, 0, 255), 2)
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
cv2.imwrite('rgb.jpg', rgb)
cv2.imwrite('hsv.jpg', hsv)
cv2.imwrite('Testimg.jpg', img)
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