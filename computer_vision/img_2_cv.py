import numpy as np
import cv2 as cv

class opencv_camera():
    def __init__(self, render, size, name, frame_interval):
        self.frame_int = frame_interval
        self.render = render   
        
        self.buffer = self.render.win.makeTextureBuffer(name, size, size, None, True)
        # self.buffer.setSort(-100)
        self.cam = self.render.makeCamera(self.buffer)
        self.cam.setName(name)     
        self.cam.node().getLens().setFilmSize(36, 24)
        self.cam.node().getLens().setFocalLength(22)
        self.name = name
        # self.render.taskMgr.add(self.set_active, name) 
        # self.render.taskMgr.add(self.set_inactive, name)
        self.buffer.setActive(1)
        self.render.graphicsEngine.renderFrame()
        
    def get_image(self):
        self.render.graphicsEngine.renderFrame()
        tex = self.buffer.getTexture()  
        img = tex.getRamImage()
        image = np.frombuffer(img, np.uint8)

        if len(image) > 0:
            image = np.reshape(image, (tex.getYSize(), tex.getXSize(), 4))
            image = cv.flip(image, 0)

            return True, image
        else:
            return False, None
    
    def set_active(self, task):
        if task.frame % 10 == 0:
            self.buffer.setActive(1)
        return task.cont
    
    def set_inactive(self, task):
        if task.frame % 10 == 1:
            self.buffer.setActive(0)
        return task.cont