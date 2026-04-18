import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Convert gray and edges to 3 channels so we can stack
    gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Resize all to same size (important)
    frame = cv2.resize(frame, (400, 300))
    gray_3 = cv2.resize(gray_3, (400, 300))
    edges_3 = cv2.resize(edges_3, (400, 300))

    # Stack horizontally
    combined = np.hstack((frame, gray_3, edges_3))

    cv2.imshow("Multi View", combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()