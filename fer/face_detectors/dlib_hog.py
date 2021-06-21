import dlib
import cv2


print("[INFO] loading face detector...")
detector = dlib.get_frontal_face_detector()


def _convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)


def detect_face(frame, use_gray=False):

    if use_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return detector(frame, 1) # return rects

def draw_boxes(rects, frame):
    boxes = [_convert_and_trim_bb(frame, r) for r in rects]

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

