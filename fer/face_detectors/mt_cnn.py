from mtcnn import MTCNN
from imutils.video import VideoStream
import imutils
import time
import cv2

print("[INFO] loading face detector...")
detector = MTCNN()


print("[INFO] starting video stream...")
video = VideoStream(src=2).start()
time.sleep(2.0)


while True:
    # check, frame = video.read()
    frame = video.read()
    frame = imutils.resize(frame, width=500)
    faces = detector.detect_faces(frame)

    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

        confidence = result['confidence']
        text = "{:.2f}%".format(confidence * 100)
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
        )

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
video.stop()
