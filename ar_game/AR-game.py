#get the corner tags.
#stretch the image
#Targets fall down, and you have to stop them.

import cv2
import cv2.aruco as aruco
import sys
import pyglet
from Game import Game

video_id = 0
shadow_mode= False
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
Left = True
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])
if len(sys.argv) > 2:
    if sys.argv[2] == "Left":
        #play with a harsh black and white background image
        Left = True
    elif sys.argv[2] == "Right":
        Left = False
if len(sys.argv) > 3:
    if int(sys.argv[3]) == 1:
        #play with a harsh black and white background image
        shadow_mode= True

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
game = Game(WINDOW_WIDTH,WINDOW_HEIGHT,window,cap,detector,shadow_mode,Left)


@window.event
def on_draw():
    window.clear()
    game.draw()
    
    pass
def update(dt):
    #spawn enemies
    game.update()
 
    pass
@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.SPACE:
        game.start_game()


pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()

cap.release()
cv2.destroyAllWindows()


#get collision map
#ballons that move one direction
    #check every frame if intersecting between black and circles.


#DONE
#development image input
