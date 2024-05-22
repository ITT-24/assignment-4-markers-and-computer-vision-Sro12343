import cv2
import os


current_dir = os.path.dirname(__file__)
image_p = os.path.join(current_dir,'sample_image.jpg')
img = cv2.imread(image_p)
WINDOW_NAME = 'Preview Window'


cv2.namedWindow(WINDOW_NAME)

def mouse_callback(event, x, y, flags, param):
    global img

    if event == cv2.EVENT_LBUTTONDOWN:
        img = cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(WINDOW_NAME, img)

cv2.imshow(WINDOW_NAME, img)

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)


cv2.waitKey(0)
