import numpy as np
import cv2 as cv

class get_image_cv():
    def __init__(self, render):
        self.render = render   
        window_size = (self.render.win.getXSize(), self.render.win.getYSize())     
        self.render.buffer = self.render.win.makeTextureBuffer('Buffer', *window_size, None, True)
        self.render.cam_1 = self.render.makeCamera(self.render.buffer)
        self.render.cam_1.setName('cam_1')     
        self.render.cam_1.node().getLens().setFilmSize(36, 24)
        self.render.cam_1.node().getLens().setFocalLength(45)

        
    def get_image(self):
        tex = self.render.buffer.getTexture()  
        img = tex.getRamImage()
        image = np.frombuffer(img, np.uint8)
        
        if len(image) > 0:
            image = np.reshape(image, (tex.getYSize(), tex.getXSize(), 4))
            image = cv.resize(image, (0,0), fx=0.5, fy=0.5)
            return True, image
        else:
            return False, None