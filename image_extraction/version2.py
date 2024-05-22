import cv2
import numpy as np
import pyglet
from pyglet.window import mouse
from PIL import Image
import sys
import os

    
current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir,'sample_image.jpg')
output_path = current_dir
resolution_x = 100
resolution_y = 100

if len(sys.argv) > 4:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    resolution_x = int(sys.argv[3])
    resolution_y = int(sys.argv[4])
    print("resize")
    #print("Length: " + str(len(sys.argv) ))
    #pass
else:
    print("Not enough entries")
#Made by Chatgpt
def pyglet_image_to_pil_image(pyglet_image):
    """Convert pyglet image to PIL image."""
    pil_image = Image.frombytes('RGBA', 
                                (pyglet_image.width, pyglet_image.height),
                                pyglet_image.get_image_data().get_data('RGBA', pyglet_image.width * 4))
    return pil_image
#Made by Chatgpt
def pil_image_to_pyglet_image(pil_image):
    """Convert PIL image to pyglet image."""
    raw_image = pil_image.tobytes()
    pyglet_image = pyglet.image.ImageData(pil_image.width, pil_image.height, 'RGBA', raw_image)
    return pyglet_image


def create_copy(img):
    pil_img= pyglet_image_to_pil_image(img)
    pil_img_copy = pil_img.copy()
    copy_img = pil_image_to_pyglet_image(pil_img_copy)
    return copy_img




source_img = cv2.imread(input_path)
edit_img = source_img.copy()  #create_copy(source_img)
height, width = source_img.shape[:2]
point_list = []


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

# Create a video capture object for the webcam
#cap = cv2.VideoCapture(video_id)

WINDOW_WIDTH = width
WINDOW_HEIGHT = height

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

@window.event
def on_draw():
    global edit_img
    window.clear()
    #ret, frame = cap.read()
    show_img = cv2glet(edit_img, 'BGR')
    show_img.blit(0, 0, 0)
    
    
@window.event
def on_mouse_press(x, y, button, modifiers):
    global edit_img
    global point_list
    if button == mouse.LEFT:
        #add positions to the array
        #add circle to image
        if len(point_list) < 4:
            point_list.append((x,height-y))
            edit_img = cv2.circle(edit_img, (x, height-y), 5, (255, 0, 0), -1)
        if len(point_list) >= 4:
            
            
            point_array = np.array(point_list, dtype=np.float32)
            point_array = order_points(point_array)
            print("Data type of point_array:", point_array.dtype)
            print(point_array)
            destionation = np.float32(np.array([[0, 0], [resolution_x, 0], [resolution_x, resolution_y], [0, resolution_y]]))
            window.set_size(resolution_x,resolution_y)
            print("destination")
            print(destionation)
            mat = cv2.getPerspectiveTransform(point_array, destionation)
            edit_img = cv2.warpPerspective(source_img,mat,(width - 1, height - 1))
            edit_img = edit_img[0:resolution_y,0:resolution_x]
            #transform image
            pass
        #add number to image

@window.event
def on_key_press(symbol, modifiers):
    global edit_img
    if symbol == pyglet.window.key.ESCAPE:
        #empty point list
        point_list.clear()
        edit_img = source_img.copy() 
        window.set_size(WINDOW_WIDTH,WINDOW_HEIGHT)
        #reset image
        return pyglet.event.EVENT_HANDLED
    if symbol == pyglet.window.key.S:
        if len(point_list) >= 4:
            cv2.imwrite(output_path, edit_img)
        pass 
    


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



#Created with chatgpt
def order_points(pts):
    
    print("test")
    print(pts)
    rect = np.zeros((4, 2), dtype="float32")

    # Calculate the centroid
    centroid = np.mean(pts, axis=0)
    print(centroid)
    
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
    
def arrange_corner_points(original_array):
    #setup the array of the screens corners.
    corners = np.float32(np.array([[0, 0], [WINDOW_WIDTH, 0], [WINDOW_WIDTH, WINDOW_HEIGHT], [0, WINDOW_HEIGHT]]))
    
    #make a copy of the existing array for edditing
    edditing_array = original_array.copy()
    
    #make an empty array to save the new point order.
    new_order_array = []
    
                       
    #Check for the first window corner which point is nearest.
    #Save that one in a new array and delete it in the eddit array
    #repeat with all the window corners.
    print("start")
    print(original_array)
    for c in corners:
        closest_point = []
        #setup initial closest point
        
    
        shortest_distance = WINDOW_WIDTH + WINDOW_HEIGHT
        for p in edditing_array:
            #get distance 
            dx = np.abs(p[0]-c[0])
            dy =  np.abs(p[0]-c[0])
            d = dx +dy
            #if dinstance is shorter than last closest point then take it as closest point
            if d <= shortest_distance:
                shortest_distance = d
                closest_point = p
                
        print(closest_point)
        new_order_array.append(closest_point)
        edditing_array.remove(p)
        
    print("end")
    print(new_order_array)
    return new_order_array
    #return new array.
    
    
    pass

pyglet.app.run()





#Implement:
#Order Clicked points based on Window corners.
#Save with S
#Resize window     DONE



#Points with numbers

#Find Paper


#DONE:
#command line parameters for: file path input, output, resolution 
#Pyglet window
#ESC Retry DONE
#Stretch
