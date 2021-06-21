from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("[INFO] starting video stream...")
# video = cv2.VideoCapture(2)
# initialize video and allow the camera sensor to warm up
video = VideoStream(src=2).start()
time.sleep(2.0)


while True:
    # check, frame = video.read()
    frame = video.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # perform face detection
    rects = detector.detectMultiScale(
        gray, 
        scaleFactor=1.05,	
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # loop over the bounding boxes
    for (x, y, w, h) in rects:
        # draw the face bounding box on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
video.stop()

