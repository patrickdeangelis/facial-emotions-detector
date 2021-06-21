import time
import logging

from imutils.video import VideoStream
import imutils
import cv2

from face_detectors.dnn import detect_face, draw_boxes


logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)


def main():
    logging.info("starting video stream...")
    video = VideoStream(src=2).start()
    time.sleep(2.0)


    while True:
        frame = video.read()
        frame = imutils.resize(frame, width=800)
        rects = detect_face(frame)

        draw_boxes(rects, frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    video.stop()


if __name__ == '__main__':
    main()

