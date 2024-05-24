from Player import Player
from Ball import Ball
import numpy as np
import os
import cv2
import pyglet
from PIL import Image
from image_calculation import image_calculation

class Game():
    def __init__(self,WINDOW_WIDTH,WINDOW_HEIGHT,window,cap,detector,shadow_mode,Left) -> None:
        self.player = Player(40)
        self.ball_countdown=30
        self.ball_list = []
        self.points = []
        self.current_dir = os.path.dirname(__file__)
        self.input_path = os.path.join(self.current_dir,'test.jpg')
        self.source_img = cv2.imread(self.input_path)
        self.image = None
        self.WINDOW_WIDTH = WINDOW_WIDTH
        self.WINDOW_HEIGHT = WINDOW_HEIGHT
        self.shadow_mode = shadow_mode
        self.game_over_text = pyglet.text.Label(text="GAME OVER", x=self.WINDOW_WIDTH /2, y=self.WINDOW_HEIGHT /2,font_size=70,color=(255,0,0,255), anchor_x='center',anchor_y='center')
        self.restart_text = pyglet.text.Label(text="Space", x=self.WINDOW_WIDTH /2, y=100,font_size=20,color=(255,0,0,255), anchor_x='center',anchor_y='center')
        self.points_text = pyglet.text.Label(text="-", x=self.WINDOW_WIDTH /2, y=100,font_size=20,color=(255,0,0,255), anchor_x='center',anchor_y='center')
        self.game_over = False
        self.immg_calc=image_calculation(cap,window,detector)
        self.Left = Left
        self.points = 0
        self.timerMax = 60
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
    
    
    def start_game(self):
        self.game_over = False
        
        
    def stop_game(self):
        self.game_over_text.x = self.immg_calc.size_x/2
        self.game_over_text.y = self.immg_calc.size_y/2
        self.restart_text.x = self.immg_calc.size_x/2
        self.points_text.text = "Your points "+ str(self.points)
        self.points_text.x = self.immg_calc.size_x/2
        self.points_text.y = self.immg_calc.size_y/2-60
        self.game_over = True 
        self.ball_list.clear()
        
        
    def draw(self):
        #capture webcam image and resize it so the game board fits.
        captured_img = self.immg_calc.capture_frame()
        if captured_img is not None:           
            show_img = captured_img
            thresh,cx,cy = self.immg_calc.calc_cursor_position(show_img)
            if self.game_over:
                #Game over screen in Black and white
                show_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                show_img = self.cv2glet(show_img, 'BGR')
                show_img.blit(0, 0, 0)
                self.game_over_text.draw()
                self.restart_text.draw()
                self.points_text.draw()
            else:
                #draw game
                if self.shadow_mode:
                    #black and white background
                    show_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                show_img = self.cv2glet(show_img, 'BGR')
                show_img.blit(0, 0, 0)
                for b in self.ball_list:
                    b.draw()
                self.player.draw()
                self.player.update(cx,self.immg_calc.size_y-cy)
                
    
    
    def update(self):
        if self.game_over != True:
            self.ball_spawner()
            #move enemies
            for b in self.ball_list:
                #update ball position
                b.update()
                #despawn old balls
                if b.despawn:
                    self.ball_list.remove(b)
                    self.points +=1
                    if self.points % 6 == 0 and self.timerMax > 15:
                        self.timerMax -= 1
        
                    continue      
                #check enemy player colision
                if self.check_collision(self.player.x,self.player.y,self.player.radius,b.x,b.y,b.radius):
                    self.stop_game()
                    break
            
        
        
    def check_collision(self,px,py,pr,bx,by,br):
        distance = np.sqrt((px - bx)**2 + (py - by)**2)
        if distance < pr+br:
            return True
        return False
        
        
    def ball_spawner(self):
        #if countdown reaches 0 spawn a ball and set new random range countdown
        self.ball_countdown -= 1

            
        if self.ball_countdown <= 0:
            radius = 10
            
            #spawn ball on the right or left.
            if self.Left:
                speed = -10
                spawn_x = self.immg_calc.size_x+10
                despawn_x = -100
            else:
                speed = 10
                spawn_x = -10
                despawn_x= self.immg_calc.size_x+100

            starting_y =  np.random.randint(0, self.immg_calc.size_y)
            self.ball_list.append(Ball(spawn_x,starting_y,speed,radius,despawn_x))
            self.ball_countdown =  np.random.randint(10, self.timerMax)