from pyglet import shapes
from pyglet.gl import glGetError, GL_NO_ERROR
import pyglet

class Player():
    def __init__(self):
        self.shape = shapes.Circle(100, 100, 40, color=(255, 0, 0, 255))
    
    
    def draw(self):      
        
        self.shape.draw()
        err = glGetError()
        if err != GL_NO_ERROR:
            raise pyglet.gl.lib.GLException(f'OpenGL error: {err}')
        pass
    
    def update(self,x,y):
        self.shape.x = x
        self.shape.y = y
        pass    
