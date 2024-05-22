import cv2
import numpy as np
import pyglet
from pyglet.window import mouse
from PIL import Image
import sys
import os


#setup variables    
current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir,'sample_image.jpg')
output_path = os.path.join(current_dir,'result_image.jpg')
resolution_x = 100
resolution_y = 100
point_list = []

#check Command line parameters
if len(sys.argv) > 4:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    resolution_x = int(sys.argv[3])
    resolution_y = int(sys.argv[4])
else:
    print("Not enough entries: Using ./sample_image.jpg ./result_image.jpg x=100 y=100")

#Load image
source_img = cv2.imread(input_path)

print(f"Trying to load image from: {input_path}")
edit_img = source_img.copy()  
height, width = source_img.shape[:2]

#Setup Pyglet Window
WINDOW_WIDTH = width
WINDOW_HEIGHT = height
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)


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

@window.event
def on_draw():
    #global edit_img
    window.clear()
    show_img = cv2glet(edit_img, 'BGR')
    show_img.blit(0, 0, 0)
    
    
#React to mouse clicks
@window.event
def on_mouse_press(x, y, button, modifiers):
    global edit_img
    global point_list
    if button == mouse.LEFT:
        if len(point_list) < 4:
            #Not enough points jet
            point_list.append((x,height-y))
            edit_img = cv2.circle(edit_img, (x, height-y), 5, (255, 0, 0), -1)
        if len(point_list) == 4:
            #Enough points so warp image
            point_array = np.array(point_list, dtype=np.float32)
            point_array = order_points(point_array)
            destionation = np.float32(np.array([[0, 0], [resolution_x, 0], [resolution_x, resolution_y], [0, resolution_y]]))
            mat = cv2.getPerspectiveTransform(point_array, destionation)
            edit_img = cv2.warpPerspective(source_img,mat,(width - 1, height - 1))
            edit_img = edit_img[0:resolution_y,0:resolution_x]
            
            #resize to result resolution
            window.set_size(resolution_x,resolution_y)

#React to Keyboard-presses
@window.event
def on_key_press(symbol, modifiers):
    global edit_img
    if symbol == pyglet.window.key.ESCAPE:
        #ESCAPE pressed so Reset Image
        #empty point list
        point_list.clear()
        edit_img = source_img.copy() 
        window.set_size(WINDOW_WIDTH,WINDOW_HEIGHT)
        #reset image
        return pyglet.event.EVENT_HANDLED
    
    if symbol == pyglet.window.key.S:
        #S pressed so if image was warped save image
        if len(point_list) >= 4:
            cv2.imwrite(output_path, edit_img)
            print("Saved to: " + output_path)
        pass 
    

#Order the Points based on their position relative to their collective center
#Created with chatgpt
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # Calculate the centroid
    centroid = np.mean(pts, axis=0)
    # Assign points based on their relative position to the centroid
    for point in pts:
        if point[0] < centroid[0] and point[1] < centroid[1]:
            rect[0] = point  # Top-left
        elif point[0] > centroid[0] and point[1] < centroid[1]:
            rect[1] = point  # Top-right
        elif point[0] > centroid[0] and point[1] > centroid[1]:
            rect[2] = point  # Bottom-right
        elif point[0] < centroid[0] and point[1] > centroid[1]:
            rect[3] = point  # Bottom-left
    return rect
    
    
pyglet.app.run()