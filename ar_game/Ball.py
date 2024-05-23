from pyglet import shapes
from pyglet.gl import glGetError, GL_NO_ERROR
import pyglet
import numpy as np

class Ball():
    def __init__(self,starting_x, starting_y, speed, radius, despawn_x):
        self.radius = radius
        self.shape = shapes.Circle(100, 100, self.radius, color=(255, 0, 0, 255))
        self.x = starting_x
        self.y = starting_y
        self.speed = speed
        self.despawn_x = despawn_x
        self.despawn = False
        
    def draw(self):      
        if self.despawn is False:
            self.shape.draw()
            err = glGetError()
            if err != GL_NO_ERROR:
                raise pyglet.gl.lib.GLException(f'OpenGL error: {err}')
            pass
    
    def update(self):
        self.x += self.speed
        if np.abs(self.x - self.despawn_x)<20:
            self.despawn = True
        
        self.shape.x = self.x
        self.shape.y = self.y
        pass    
