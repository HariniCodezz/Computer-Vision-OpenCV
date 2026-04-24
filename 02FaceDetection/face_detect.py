import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("Error loading cascade file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (IMPORTANT: use gray, not BGR)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles on BGR frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Optional: create gray & edge views
    edges = cv2.Canny(gray, 100, 200)

    gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Draw rectangles on all views (visual consistency)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray_3, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(edges_3, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Resize
    frame = cv2.resize(frame, (400, 300))
    gray_3 = cv2.resize(gray_3, (400, 300))
    edges_3 = cv2.resize(edges_3, (400, 300))

    combined = np.hstack((frame, gray_3, edges_3))

    cv2.imshow("Face Detection (BGR View)", combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()