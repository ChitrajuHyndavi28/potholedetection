import cv2

cap = cv2.VideoCapture(0)  # try 1 or 2 if 0 doesn't work

if not cap.isOpened():
    print("Cannot open webcam")
else:
    print("Webcam opened successfully")

ret, frame = cap.read()
if ret:
    print("Frame grabbed successfully")
else:
    print("Failed to grab frame")

cap.release()
