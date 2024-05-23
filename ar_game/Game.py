from Player import Player
from Ball import Ball
import numpy as np
import os
import cv2
import pyglet
from PIL import Image


class Game():
    def __init__(self,WINDOW_WIDTH,WINDOW_HEIGHT,window,cap,detector,shadow_mode) -> None:
        self.player = Player(40)
        self.ball_countdown=30
        self.ball_list = []
        self.use_old_frame = 0
        self.points = []
        self.current_dir = os.path.dirname(__file__)
        self.input_path = os.path.join(self.current_dir,'test.jpg')
        self.source_img = cv2.imread(self.input_path)
        self.image = None
        self.WINDOW_WIDTH = WINDOW_WIDTH
        self.WINDOW_HEIGHT = WINDOW_HEIGHT
        self.window = window
        self.cap = cap
        self.size_x = None
        self.size_y = None
        self.detector = detector
        self.shadow_mode = shadow_mode
        pass
    
    
    # converts OpenCV image to PIL image and then to pyglet texture
    # https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
    def cv2glet(self,img,fmt):
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
    
    
    def draw(self):
        captured_img = self.capture_frame()
        if captured_img is not None:
            show_img = captured_img
            thresh,cx,cy = self.calc_cursor_position(show_img)
            
            if self.shadow_mode:
                show_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            show_img = self.cv2glet(show_img, 'BGR')
            show_img.blit(0, 0, 0)
            self.player.draw()
            self.player.update(cx,self.WINDOW_HEIGHT-cy)
            
            for b in self.ball_list:
                b.draw()
                
    def update(self):
        self.ball_spawner()
        #move enemies
        for b in self.ball_list:
            #update ball position
            b.update()
            if b.despawn:
                self.ball_list.remove(b)
                continue      
            #check enemy player colision
            if self.check_collision(self.player.x,self.player.y,self.player.radius,b.x,b.y,b.radius):
                print("Game Over")
                pass
            
        
    def check_collision(self,px,py,pr,bx,by,br):
        distance = np.sqrt((px - bx)**2 + (py - by)**2)
        if distance < pr+br:
            return True
        return False
        
        
    def ball_spawner(self):
        self.ball_countdown -= 1
        if self.ball_countdown <= 0:
            l_or_r = np.random.choice([True, False])
            
            radius = 10
            if l_or_r:
                speed = 10
                spawn_x = -10
                despawn_x= self.WINDOW_WIDTH+100
            else:
                speed = -10
                spawn_x = self.WINDOW_WIDTH+10
                despawn_x = -100
                
            starting_y =  np.random.randint(0+100, self.WINDOW_HEIGHT-100)
            self.ball_list.append(Ball(spawn_x,starting_y,speed,radius,despawn_x))
            self.ball_countdown =  np.random.randint(10, 60)
        
    
    #touch to destroy some ballones but not others
    pass

    def calc_cursor_position(self,frame):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Calculate mean brightness of the grayscale image
        mean_brightness = img_gray.mean()

        # Adjust the cutoff point based on mean brightness
        cutoff = int(mean_brightness * 0.7) 
        block_size = 25  #11
        C_value = 10   #2

        #The idea to also use the saturation of the hand to set it apart from the paper is based on a conversation with Maximilian Kilger
        ret,thresh = cv2.threshold(img_gray, cutoff, 255, cv2.THRESH_BINARY)#cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C_value)
        invert_thresh = cv2.bitwise_not(thresh)
        
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        sat_threshold = 80
        _, s_thresh = cv2.threshold(s, sat_threshold, 255, cv2.THRESH_BINARY)          
        s_invert_thresh = cv2.bitwise_not(thresh)
        
        combined_thresh = cv2.bitwise_and(s_invert_thresh, invert_thresh)
        
        
       
        #Getting the position based on the contour was made with Copilot 
        contours, hierarchy = cv2.findContours(combined_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

    def get_interaction_image(self,frame):
        cutoff = 128
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, cutoff, 255, cv2.THRESH_BINARY)
        return thresh


    def capture_frame(self):

        #depuging work
        #frame = source_img.copy()
        
        # Capture a frame from the webcam
        ret, frame = self.cap.read()

        if self.size_x == None:
            #print(type(frame))
            self.size_y, self.size_x, channels = frame.shape
            self.window.set_size(self.size_x,self.size_y)
            self.destionation = np.float32(np.array([[0,0], [self.size_x, 0], [self.size_x, self.size_y], [0, self.size_y] ]))
            self.points = self.destionation

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the frame
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)


        #If marker detection not succeding continue using old marker position for perspective transform
        #It used to only use old marker positions for 3 frames, but after talking to Maximilian Kilger i changed it to this.
        if ids is not None:        
            if len(ids) >=4:
                print("1")
                self.set_points(corners)
    
        self.use_old_frame -=1
        mat = cv2.getPerspectiveTransform(self.points, self.destionation)
        edit_img = cv2.warpPerspective(frame,mat,(self.size_x, self.size_y))
        return edit_img




    def set_points(self,corners):
        self.points = self.get_marker_centers(corners)
        self.points = self.order_points(self.points)
        self.points = np.array(self.points, dtype=np.float32)
        #return self.points
        
        
        
        
    def get_marker_centers(self,corners):
        centers = []

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
    def order_points2(self,pts):
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
    def order_points(self,pts):
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
