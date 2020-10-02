# BASIC LIBRARIES
import torch
import cv2 as cv
import numpy as np
import time
from panda3d.core import Thread

# LANDING SETUP
from visual_landing.ppo_aux import PPO
from visual_landing.landing_reward_fuction import visual_reward
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TIME_STEP = 0.01
IMAGE_LEN = np.array([160, 160])
            
# LANDING POLICY

VELOCITY_SCALE = [1, 1, 4]


class ppo_worker():
    def __init__(self, render, quad_workers, cv_cam, ldg_policy):
        
        # MODELS AND RENDER SETUP
        self.render = render
        self.quad_model = render.quad_model
        self.prop_models = render.prop_models
        self.quad_workers = quad_workers
        self.a = np.zeros(4)
        self.ldg_policy = ldg_policy
        self.cv_cam = cv_cam
        
    def render_position(self, coordinates, marker_position):

        self.render.checker.setPos(*tuple(marker_position))
        
        pos = coordinates[0:3]
        ang = coordinates[3:6]
        w = coordinates[6::]
        
        for i, w_i in enumerate(w):
            self.a[i] += (w_i*TIME_STEP)*180/np.pi/10
        ang_deg = (ang[2]*180/np.pi, ang[0]*180/np.pi, ang[1]*180/np.pi)
        pos = (0+pos[0], 0+pos[1], 5+pos[2])
        
        self.quad_model.setPos(*pos)
        self.quad_model.setHpr(*ang_deg)

        for prop, a in zip(self.prop_models, self.a):
            prop.setHpr(a, 0, 0)

        
    def take_picture(self):
        ret, image = self.cv_cam.get_image() 
        if ret:
            return cv.cvtColor(image[:,:,0:4], cv.COLOR_BGR2GRAY)
        else:
            return None
                
        
    def wait_until_ready(self, task):        
        for quad_worker in self.quad_workers:
            if quad_worker.update[0] and not quad_worker.visual_done:
                self.step(quad_worker) 
                quad_worker.update = [0, None]
                quad_worker.wait_for_task = False 
        return task.cont
        
        
    def step(self, quad_worker):                   

        coordinates = quad_worker.update[1]

        quad_worker.reward, quad_worker.last_shaping, quad_worker.visual_done = visual_reward(quad_worker.marker_position, quad_worker.quad_env.state[0:5:2], quad_worker.quad_env.state[1:6:2], quad_worker.vel_error, quad_worker.last_shaping)
        
        self.render_position(coordinates, quad_worker.marker_position)
        
        image = self.take_picture()

        quad_worker.image_roll(image)  

        if quad_worker.ppo_calls >= 3:
        
            quad_worker.memory.rewards.append(quad_worker.reward)
            
            quad_worker.memory.is_terminals.append(quad_worker.visual_done)    
            
            network_in = np.array([quad_worker.image_1, quad_worker.image_2, quad_worker.image_3])
            
            visual_action = self.ldg_policy.select_action(network_in, quad_worker.memory)
            
            quad_worker.vel_error = visual_action*VELOCITY_SCALE            
        
        quad_worker.ppo_calls += 1