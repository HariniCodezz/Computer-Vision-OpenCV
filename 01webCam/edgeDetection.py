import cv2
capture=cv2.VideoCapture(0)
cv2.namedWindow("Canny_Edge_Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Canny_Edge_Detection", 1200, 900)
while True:
    ret, frame=capture.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray, 79, 200)
    
    cv2.imshow("Canny_Edge_Detection", edges)
    if cv2.waitKey(1)==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()