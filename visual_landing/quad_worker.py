# BASIC LIBRARIES
import torch
import sys
import numpy as np
import cv2 as cv
import time
# ENVIRONMENT AND CONTROLLER SETUP
from environment.quadrotor_env_opt import quad
from environment.controller.model import ActorCritic
from environment.controller.dl_auxiliary import dl_in_gen
from panda3d.core import Thread

from visual_landing.memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

T = 5
TIME_STEP = 0.01
TOTAL_STEPS = 1500
IMAGE_LEN = np.array([160, 160])
TASK_INTERVAL_STEPS = 10


#CONTROL POLICY
AUX_DL = dl_in_gen(T, 13, 4)
state_dim = AUX_DL.deep_learning_in_size
CRTL_POLICY = ActorCritic(state_dim, action_dim=4, action_std=0).to(device)
try:
    CRTL_POLICY.load_state_dict(torch.load('./environment/controller/PPO_continuous_solved_drone.pth',map_location=device))
    print('Saved Control policy loaded')
except:
    print('Could not load Control policy')
    sys.exit(1)  
  
           
class quad_worker():
    def __init__(self, render):
        self.update = [0, None]
        self.done = False
        self.render = render
        
        #CONTROLLER POLICY
        self.quad_env = quad(TIME_STEP, TOTAL_STEPS, 1, T)
        states, action = self.quad_env.reset(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
        self.aux_dl =  dl_in_gen(T, 13, 4)
        self.network_in = self.aux_dl.dl_input(states, action)
        self.image_zeros() 
        self.memory = Memory()
        
        #TASK MANAGING
        self.wait_for_task = False
        self.visual_done = False
        self.ppo_calls = 0
        
        #LDG TRAINING
        self.vel_error = np.zeros(3)
        self.last_shaping = None
        
        #MARKER POSITION
        random_marker_position = np.random.normal([0, 0], 0.8)
        self.render.checker.setPos(*tuple(random_marker_position), 0.001)
        self.marker_position = np.append(random_marker_position, 0.001)
        self.internal_frame = 0
        
    def image_zeros(self):
        self.image_1 = np.zeros(IMAGE_LEN) 
        self.image_2 = np.zeros(IMAGE_LEN)
        self.image_3 = np.zeros(IMAGE_LEN)   
     
    def image_roll(self, image):
        self.image_3 = self.image_2
        self.image_2 = self.image_1
        self.image_1 = image
         
    def step(self, task):
        
        if self.visual_done:
            return task.done
        
        if not self.wait_for_task:
            for i in range(TASK_INTERVAL_STEPS):
                
                self.internal_frame += 1
                # LOWER CONTROL STEP  
    
                crtl_action = CRTL_POLICY.actor(torch.FloatTensor(self.network_in).to(device)).cpu().detach().numpy()
    
                states, _, done = self.quad_env.step(crtl_action)
                
                # CONTROL DIFFERENCE
                error_smoothing = self.vel_error/TASK_INTERVAL_STEPS*(self.internal_frame%TASK_INTERVAL_STEPS)
    
                state_error = np.zeros([1,14])
                state_error[0, 1] = self.vel_error[0]
                state_error[0, 3] = self.vel_error[1]
                state_error[0, 5] = self.vel_error[2]
                
                # CONTROL DIFFERENCE
                error_smoothing = states + state_error/TASK_INTERVAL_STEPS*(self.internal_frame%TASK_INTERVAL_STEPS)
                
                self.network_in = self.aux_dl.dl_input(error_smoothing, [crtl_action])
                        
                coordinates = np.concatenate((states[0, 0:5:2], self.quad_env.ang, np.zeros(4))) 
                
                Thread.considerYield()
                
            self.update = [1, coordinates]
            self.wait_for_task = True

            return task.cont 
        else:
            return task.cont