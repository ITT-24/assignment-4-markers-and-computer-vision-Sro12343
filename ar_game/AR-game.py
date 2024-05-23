#get the corner tags.
#stretch the image
#Targets fall down, and you have to stop them.


import cv2
import cv2.aruco as aruco
import sys
import pyglet
from pyglet.window import mouse
import numpy as np

import os



from Game import Game

video_id = 0
shadow_mode= False
if len(sys.argv) > 2:
    video_id = int(sys.argv[1])
    if int(sys.argv[2]) == 1:
        #play with a harsh black and white background image.
        shadow_mode= True


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)







WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

size_x = None 
size_y = None




window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
game = Game(WINDOW_WIDTH,WINDOW_HEIGHT,window,cap,detector,shadow_mode)


@window.event
def on_draw():
    window.clear()
    game.draw()
    
    pass
def update(dt):
    #spawn enemies
    game.update()
 
    pass



pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()

cap.release()
cv2.destroyAllWindows()


#get collision map
#ballons that move one direction
    #check every frame if intersecting between black and circles.


#DONE
#development image input