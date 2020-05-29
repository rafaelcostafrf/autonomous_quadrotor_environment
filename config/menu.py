from direct.gui.OnscreenText import OnscreenText
from direct.gui import *
from direct.gui.DirectGui import *
from panda3d.core import *
import numpy as np

class menu():
    
    def __init__(self, env, drone, sensor, cv):
        # Add some text
        frame = DirectFrame(frameSize = (-0.27,0.27,-0.30,0.26), frameColor = (0.35, 0.35, 0.35, 0.55), pos = (1.5, 0, 0))
        self.drone = drone
        bk_text = "Alfa Version: 0.0.1"
        OnscreenText(text=bk_text, pos=(1.5, -0.24), scale=0.04,
                    fg=(1, 1, 1, 1), shadow=(0,0,0,1), align=TextNode.ACenter,
                    mayChange=0)
        
        # Add some text
        string_1 = 'Real State' if drone.REAL_CTRL else ('Hybrid' if cv.IMG_POS_DETER else 'MEMS') 
        
        output = 'Operation Mode: '+string_1
        textObject = OnscreenText(text=output, pos=(1.5, -0.18), scale=0.04,
                                  fg=(1, 1, 1, 1), shadow=(0,0,0,1), align=TextNode.ACenter,
                                  mayChange=1)
        
        # Callback function to set  text
        def true_state():
            textObject.text=('Operation Mode: True State')
            self.drone.REAL_CTRL = True
            cv.IMG_POS_DETER = False
            self.drone.error=[]
            
        def mems():    
            textObject.text=('Operation Mode: Mems')
            drone.REAL_CTRL = False
            cv.IMG_POS_DETER = False
            self.drone.error=[]
            
        def hybrid():
            textObject.text=('Operation Mode: Hybrid')
            self.drone.REAL_CTRL = False
            cv.IMG_POS_DETER = True
            self.drone.error=[]
        
        a = DirectButton(text=("True State"), borderWidth=(.3, .3), pos=(0,0,0), scale= 0.05, command=true_state)
        a.setPos((1.5, 0, 0))

        b = DirectButton(text=("Mems"), borderWidth=(.3, .3), scale = 0.05, command=mems)
        
        b.setPos((1.5, 0, 0.08))
        c = DirectButton(text=("Hybrid"), borderWidth=(.3, .3), scale = 0.05, command=hybrid)        
        c.setPos((1.5, 0, -0.08))        
        env.taskMgr.add(self.menu_update, 'Menu Update')  
        
        
        
    def menu_update(self, task):
        if task.frame == 0:
            self.error_text = OnscreenText(text='', pos=(1.5, 0.16), scale=0.04,
                                  fg=(1, 1, 1, 1), shadow=(0,0,0,1), align=TextNode.ACenter,
                                  mayChange=1)
        if task.frame % 30 == 0:
            if len(self.drone.error) > 1:
                error = np.sum(np.square(self.drone.error))
            else:
                error = 0
            error_str = f'Error: {error:.5f} '
            self.error_text.text = error_str
        return task.cont