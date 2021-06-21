# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os.path


# load our serialized model from disk
PROTO_TXT = "deploy.prototxt.txt"
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
MIN_CONFIDENCE = 0.5
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROTO_TXT_PATH = os.path.join(BASE_PATH, PROTO_TXT)
MODEL_PATH = os.path.join(BASE_PATH, MODEL)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(
    PROTO_TXT_PATH, MODEL_PATH
)


def detect_face(frame):

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    return net.forward() # detections


def draw_boxes(detections, frame):
    (h, w) = frame.shape[:2]

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < MIN_CONFIDENCE:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
