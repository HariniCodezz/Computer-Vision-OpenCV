import cv2
capture=cv2.VideoCapture(0) 
if not capture.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame=capture.read()
    if not ret:
        print("Failed to capture the image")
    cv2.imshow("Your screen goes here!!", frame)

    if cv2.waitKey(1) ==ord('q'):
        break;
capture.release()
cv2.destroyAllWindows()