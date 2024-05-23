import numpy as np
import cv2


class image_calculation():
    def __init__(self,cap, window,detector):
        self.cap = cap
        self.window = window
        self.detector = detector
        self.size_x = None
        self.size_y = None
        
        
    def calc_cursor_position(self,frame):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean brightness of the grayscale image
        mean_brightness = img_gray.mean()
        
        # Adjust the cutoff point based on mean brightness
        cutoff = int(mean_brightness * 0.7) 
        
        #The idea to also use the saturation of the hand to set it apart from the paper is based on a conversation with Maximilian Kilger
        ret,thresh = cv2.threshold(img_gray, cutoff, 255, cv2.THRESH_BINARY)
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
            #Check if the contour is larger than the current largest contour
            if area > largest_area:
                largest_area = area
                largest_contour = contour
        #calculate middle point of contour. ChatGPT
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            x, y, w, h = cv2.boundingRect(contour)
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        return thresh, cx,cy


    def capture_frame(self):
        # Capture a frame from the webcam
        ret, frame = self.cap.read()
        if self.size_x == None:
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
                self.set_points(corners)
        mat = cv2.getPerspectiveTransform(self.points, self.destionation)
        edit_img = cv2.warpPerspective(frame,mat,(self.size_x, self.size_y))
        return edit_img


    def set_points(self,corners):
        self.points = self.get_marker_centers(corners)
        self.points = self.order_points(self.points)
        self.points = np.array(self.points, dtype=np.float32)

    
    #Created with chatgpt
    def get_marker_centers(self,corners):
        centers = []
        for c in corners:
            if len(c[0]) != 4:
                raise ValueError("There should be exactly four corner points for each marker.")
            corners = np.array(corners, dtype=np.float32)
            # Calculate the center by averaging the x and y coordinates
            center_x = np.mean(c[0][:, 0])
            center_y = np.mean(c[0][:, 1])
            centers.append((center_x, center_y))
        return np.array(centers, dtype=np.float32)


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