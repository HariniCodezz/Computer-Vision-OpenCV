import cv2
import time

cap = cv2.VideoCapture(0)

# Set camera resolution (helps FPS)
cap.set(3, 640)
cap.set(4, 480)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("Error loading cascade file")
    exit()

prev_time = 0
frame_count = 0
faces = []

blur_on = True  # toggle

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Resize for faster detection
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    frame_count += 1

    # Detect faces every 3 frames
    if frame_count % 3 == 0:
        detected = face_cascade.detectMultiScale(gray_small, 1.3, 5)
        faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in detected]

    # Blur background (fast blur)
    blurred = cv2.blur(frame, (10, 10))
    output = blurred.copy()

    # Keep faces clear
    for (x, y, w, h) in faces:
        output[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(output, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Toggle between original and blurred
    display_frame = output if blur_on else frame

    cv2.imshow("Optimized Smart Webcam", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('b'):
        blur_on = not blur_on

cap.release()
cv2.destroyAllWindows()