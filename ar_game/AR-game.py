#get the corner tags.
#stretch the image
#Targets fall down, and you have to stop them.


import cv2
import cv2.aruco as aruco
import sys
import pyglet
from pyglet.window import mouse
import numpy as np
from PIL import Image
import os
from Player import Player

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])



aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)



# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
      rows, cols = img.shape
      channels = 1
    else:
      rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg




WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

size_x = None 
size_y = None



current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir,'test.jpg')
source_img = cv2.imread(input_path)

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

image = None


player = Player()
@window.event
def on_draw():
    window.clear()
    show_img = capture_frame()
    if show_img is not None:
        thresh,cx,cy = calc_cursor_position(show_img)
        show_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        show_img = cv2glet(show_img, 'BGR')
        show_img.blit(0, 0, 0)
        player.draw()
        player.update(cx,WINDOW_HEIGHT-cy)
    pass


def calc_cursor_position(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate mean brightness of the grayscale image
    mean_brightness = img_gray.mean()

    # Adjust the cutoff point based on mean brightness
    cutoff = int(mean_brightness * 0.7) 
    block_size = 25  #11
    C_value = 10   #2

    ret,thresh = cv2.threshold(img_gray, cutoff, 255, cv2.THRESH_BINARY)#cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C_value)
    
    invert_thresh = cv2.bitwise_not(thresh)
    #Getting the position based on the contour was made with Copilot 
    contours, hierarchy = cv2.findContours(invert_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    largest_contour = None

    # Iterate through contours
    for contour in contours:
        # Calculate area of the contour
        area = cv2.contourArea(contour)

        # Check if the contour is larger than the current largest contour
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    
    #if not contours:
    #    return thresh,0,0
    
    #largest_contour = max(contours, key=lambda x: cv2.contourArea(x))
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        x, y, w, h = cv2.boundingRect(contour)
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return thresh, cx,cy
    pass

def get_interaction_image(frame):
    cutoff = 128
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, cutoff, 255, cv2.THRESH_BINARY)
    return thresh

def check_collision():
    #for each ballon check overlap with black area of threthhold. 
    #if overlap then destroy.
    pass

def game_manager():
    #touch to destroy some ballones but not others
    pass

def capture_frame():
    global size_x
    global size_y
    
    #depuging work
    #frame = source_img.copy()
    
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if size_x == None:
        #print(type(frame))
        size_y, size_x, channels = frame.shape
        window.set_size(size_x,size_y)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    if ids is not None:
        
        if len(ids) >=4:
            return resize_frame(corners, frame,ids)

    
    return frame
    
    
def resize_frame(corners,frame,ids):
    points = []
    points = get_marker_centers(corners)
    points = order_points(points)
    points = np.array(points, dtype=np.float32)
    destionation = np.float32(np.array([[0,0], [size_x, 0], [size_x, size_y], [0, size_y] ]))
    mat = cv2.getPerspectiveTransform(points, destionation)
    edit_img = cv2.warpPerspective(frame,mat,(size_x, size_y))
    return edit_img
    
def get_marker_centers(corners):
    centers = []
    #print("center")
    #print(corners)
    for c in corners:
        if len(c[0]) != 4:
            #print(c[0])
            raise ValueError("There should be exactly four corner points for each marker.")
        
        # Convert the corners to a NumPy array for easier manipulation
        corners = np.array(corners, dtype=np.float32)

        # Calculate the center by averaging the x and y coordinates
        center_x = np.mean(c[0][:, 0])
        center_y = np.mean(c[0][:, 1])

        centers.append((center_x, center_y))

    return np.array(centers, dtype=np.float32)

#Created with chatgpt
def order_points(pts):
    # Create an array to hold the ordered points
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of the coordinates (x + y)
    s = pts.sum(axis=1)
    # Difference of the coordinates (y - x)
    diff = np.diff(pts, axis=1)

    # Top-left point has the smallest sum
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point has the largest sum
    rect[2] = pts[np.argmax(s)]
    # Top-right point has the smallest difference
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left point has the largest difference
    rect[3] = pts[np.argmax(diff)]

    return rect

pyglet.app.run()

cap.release()
cv2.destroyAllWindows()


#get collision map
#ballons that move one direction
    #check every frame if intersecting between black and circles.


#DONE
#development image input