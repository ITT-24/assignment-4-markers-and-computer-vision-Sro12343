import cv2
import os
import sys

current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir,'sample_image.jpg')
output_path = current_dir
resolution_x = 100
resolution_y = 100

if len(sys.argv) > 4:
    input_path = sys.argv[1]
    output_path = sys.argv[1]
    resolution_x = sys.argv[1]
    resolution_y = sys.argv[1]
    pass
else:
    print("Not enough entries")




img = cv2.imread(input_path)
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




#Implement: 
#Pyglet window
#Points with numbers
#ESC Retry
#Save with S
#Stretch


#DONE:
#command line parameters for: file path input, output, resolution 