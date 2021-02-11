import sys, os
import time

#Panda 3D Imports
from panda3d.core import Filename
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile, InputDevice, GamepadButton
loadPrcFile('./config/conf.prc')



import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-u', action='store', dest = 'user', help='User name', type=str, default = 'jon_doe')
results = parser.parse_args()



#Custom Functions
from panda3d.core import WindowProperties 
from visual_landing.ppo_world_setup import world_setup, quad_setup
from models.camera_control import camera_control_v2
from computer_vision.cameras_setup import cameras
from manual_flight_add.quad_controller import quad_sim
from models.camera_control import camera_control

mydir = os.path.abspath(sys.path[0])

mydir = Filename.fromOsSpecific(mydir).getFullpath()
T = 0.01
class MyApp(ShowBase):
    def __init__(self):

        ShowBase.__init__(self)     
        render = self.render
        print(self.devices.getDevices(InputDevice.DeviceClass.gamepad))
        self.gamepad = self.devices.getDevices(InputDevice.DeviceClass.gamepad)[0]      
        self.attachInputDevice(self.gamepad, prefix="gamepad")

        # MODELS SETUP
        world_setup(self, render, mydir)
        quad_setup(self, render, mydir)
        self.camera_crtl = camera_control_v2(self, self.render) 
        self.quad_sim = quad_sim(self, results.user)
        # Accept button events of the first connected gamepad
        # self.accept("gamepad-back", exit)
        # self.accept("gamepad-start", exit)
        # self.accept("gamepad-face_x", self.reset)
        # self.accept("gamepad-face_a", self.action, extraArgs=["face_a"])
        # self.accept("gamepad-face_a-up", self.actionUp)
        # self.accept("gamepad-face_b", self.action, extraArgs=["face_b"])
        # self.accept("gamepad-face_b-up", self.actionUp)
        # self.accept("gamepad-face_y", self.action, extraArgs=["face_y"])
        # self.accept("gamepad-face_y-up", self.actionUp)
      
        
        self.look_at = False
        self.last_time_press = 0
        
        self.taskMgr.add(self.controller_function, 'Controller Read')

    def controller_function(self, task):
        init_time = time.time()
        lshoulder = self.gamepad.findButton('lshoulder')
        rshoulder = self.gamepad.findButton('rshoulder')
        left_x = self.gamepad.findAxis(InputDevice.Axis.left_x).value
        left_y = self.gamepad.findAxis(InputDevice.Axis.left_y).value
        right_y = self.gamepad.findAxis(InputDevice.Axis.right_x).value
        right_x = self.gamepad.findAxis(InputDevice.Axis.right_y).value
        left_x, left_y, right_x, right_y = deadzones(left_x, left_y, right_x, right_y)
    
                
        self.quad_sim.step(right_x, right_y, left_y, lshoulder.pressed, rshoulder.pressed)
        while True:
            if time.time()-init_time >= T:
                return task.cont
           
def deadzones(a, b, c, d):
    if a < 0.1 and a > -0.1:
        a = 0 
    if b < 0.1 and b > -0.1:
        b = 0   
    if c < 0.1 and c > -0.1:
        c = 0 
    if d < 0.1 and d > -0.1:
        d = 0 
    return a, b, c, d
app = MyApp()

app.run()