
# BASIC LIBRARIES
import torch
import sys
import numpy as np
import cv2 as cv
import time
import gc
import os
import psutil
process = psutil.Process(os.getpid())
# ENVIRONMENT AND CONTROLLER SETUP
from environment.quadrotor_env_opt import quad, sensor
from environment.quaternion_euler_utility import deriv_quat
from environment.controller.model import ActorCritic
from environment.controller.dl_auxiliary import dl_in_gen
from collections import deque

from visual_landing.ppo_trainer import PPO
from visual_landing.rl_memory import Memory
from visual_landing.rl_reward_fuction import visual_reward
from visual_landing.memory_leak import debug_gpu
import matplotlib.pyplot as plt

T = 5
T_visual_time = [9, 6, 3, 2, 0]
T_visual = len(T_visual_time)
T_total = T_visual_time[0]+1
EVAL_FREQUENCY = 1

TIME_STEP = 0.01
TOTAL_STEPS = 4000

IMAGE_LEN = np.array([88, 88])
TASK_INTERVAL_STEPS = 10
BATCH_SIZE = 356
VELOCITY_SCALE = [1, 1, 1]
VELOCITY_D = [0, 0, -VELOCITY_SCALE[2]]
#CONTROL POLICY
AUX_DL = dl_in_gen(T, 13, 4)
state_dim = AUX_DL.deep_learning_in_size
CRTL_POLICY = ActorCritic(state_dim, action_dim=4, action_std=0)
try:
    CRTL_POLICY.load_state_dict(torch.load('./environment/controller/PPO_continuous_solved_drone.pth'))
    print('Saved Control policy loaded')
except:
    print('Could not load Control policy')
    sys.exit(1)  

PLOT_LENGTH = 100
CONV_SIZE = 256
       
class quad_worker():
    def __init__(self, render, cv_cam, child_number = None, child = False):
        self.update = [0, None]
        self.done = False
        self.render = render
        self.cv_cam = cv_cam
        self.ldg_policy = PPO(3, child, T_visual)
        self.train_time = False

        self.child = child
        if child:
            self.child_number = child_number
            self.device = torch.device('cpu')  
        else: 
            self.n_samples = 0
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        print(self.device)
        
        self.quad_model = render.quad_model
        self.prop_models = render.prop_models
        self.a = np.zeros(4)
        
        #CONTROLLER POLICY
        self.quad_env = quad(TIME_STEP, TOTAL_STEPS, 1, T)
        self.sensor = sensor(self.quad_env)  
        states, action = self.quad_reset_random()
        self.sensor.reset()
        self.aux_dl =  dl_in_gen(T, 13, 4)
        self.control_network_in = self.aux_dl.dl_input(states, action)
        self.image_zeros() 
        self.memory = Memory()
        
        #TASK MANAGING
        self.wait_for_task = False
        self.visual_done = False
        self.ppo_calls = 0
        self.render.taskMgr.setupTaskChain('async', numThreads = 16, tickClock = None,
                                   threadPriority = None, frameBudget = None,
                                   frameSync = None, timeslicePriority = None)
        # self.render.taskMgr.add(self.step, 'ppo step', taskChain = 'async')
        self.render.taskMgr.add(self.quad_step, 'ppo step')
        #LDG TRAINING
        self.vel_error = np.zeros([3])
        self.last_shaping = None
        
        #MARKER POSITION
        self.internal_frame = 0
        self.crtl_action = np.zeros(4)
        #CAMERA SETUP
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        self.cv_cam = cv_cam
        self.cv_cam.cam.setPos(0, 0, 0)
        self.cv_cam.cam.setHpr(0, 270, 0)
        self.cv_cam.cam.reparentTo(self.render.quad_model)
               
        self.eval_flag = True
        self.reward_accum = 0
        self.train_calls = 0
           
        self.old_conv = torch.zeros([1, CONV_SIZE]).to(self.device)
        
    def quad_reset_random(self):
        random_marker_position = np.random.normal([0, 0], 0.8)
        self.render.checker.setPos(*tuple(random_marker_position), 0.001)
        self.marker_position = np.append(random_marker_position, 0.001)
        
        
        quad_random_z = -5*np.random.random()+1
        quad_random_xy = self.marker_position[0:2]+(np.random.random(2)-0.5)*quad_random_z/7*5/1.2
        initial_state = np.array([quad_random_xy[0], 0, quad_random_xy[1], 0, quad_random_z, 0, 1, 0, 0, 0, 0, 0, 0])
        states, action = self.quad_env.reset(initial_state)
        return states, action
        
    def sensor_sp(self):
            _, self.velocity_accel, self.pos_accel = self.sensor.accel_int()
            self.quaternion_gyro = self.sensor.gyro_int()
            self.ang_vel = self.sensor.gyro()
            quaternion_vel = deriv_quat(self.ang_vel, self.quaternion_gyro)
            self.pos_gps, self.vel_gps = self.sensor.gps()
            self.quaternion_triad, _ = self.sensor.triad()
            pos_vel = np.array([self.pos_accel[0], self.velocity_accel[0],
                                self.pos_accel[1], self.velocity_accel[1],
                                self.pos_accel[2], self.velocity_accel[2]])
            states_sens = np.array([np.concatenate((pos_vel, self.quaternion_gyro, quaternion_vel))   ])
            return states_sens
            
    def image_zeros(self):
        self.images = np.zeros([T_visual_time[0]+1, IMAGE_LEN[0], IMAGE_LEN[0]])
     
    def image_roll(self, image):
        image_copy = np.copy(image)
        # std = np.std(image_copy)
        # if std > 1e-5:
        #     image_copy = (image_copy-np.mean(image_copy))/np.std(image_copy)
        # else:
            # image_copy = (image_copy-np.mean(image_copy))
        self.print_image = image_copy
        self.images = np.roll(self.images, 1, 0)
        self.images[0] = image_copy

        
        
    def take_picture(self):
        ret, image = self.cv_cam.get_image() 
        if ret:
            return cv.cvtColor(image[:,:,0:3], cv.COLOR_BGR2GRAY)/255.0
            # print(np.shape(image))
            # return np.swapaxes(image[:,:,0:3]/255.0, 0, 2)
        else:
            return np.zeros([1, IMAGE_LEN[0], IMAGE_LEN[0]])
   
    def reset(self):
        states, action = self.quad_reset_random()
        self.sensor.reset()
        self.aux_dl =  dl_in_gen(T, 13, 4)
        self.control_network_in = self.aux_dl.dl_input(states, action)
        self.visual_done = False
        self.ppo_calls = 0
        self.old_conv = torch.zeros([1, CONV_SIZE]).to(self.device)
        #LDG TRAINING
        self.vel_error = np.zeros([3])
        self.last_shaping = None
        self.crtl_action = np.zeros(4)
        #MARKER POSITION
        self.internal_frame = 0
        self.image_zeros()
        if self.train_time :
            if self.train_calls % EVAL_FREQUENCY == 0:
                self.eval_flag = True
                self.reward_accum = 0
            self.train_time = False  
        


        
        
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
        self.render.dlightNP.setPos(*pos)
        for prop, a in zip(self.prop_models, self.a):
            prop.setHpr(a, 0, 0)
        self.render.graphicsEngine.renderFrame()
    
    def quad_step(self, task):
        init_time = time.time()
        self.internal_frame += 1
        # LOWER CONTROL STEP  
        states_sens = self.sensor_sp()
        # CONTROL DIFFERENCE
        error = np.array([[0, self.vel_error[0], 0, self.vel_error[1], 0, self.vel_error[2], 0, 0, 0, 0, 0, 0, 0, 0]])

        self.control_network_in = self.aux_dl.dl_input(states_sens-error, [self.crtl_action])
        crtl_network_in = torch.FloatTensor(self.control_network_in).to('cpu')

        self.crtl_action = CRTL_POLICY.actor(crtl_network_in).cpu().detach().numpy()

        states, _, _ = self.quad_env.step(self.crtl_action)

        coordinates = np.concatenate((states[0, 0:5:2], self.quad_env.ang, np.zeros(4))) 
        self.render_position(coordinates, self.marker_position)
        if TASK_INTERVAL_STEPS-self.internal_frame <= T_visual_time[0]+2:                
            image = self.take_picture()
            self.image_roll(image)
        if self.internal_frame == TASK_INTERVAL_STEPS or self.quad_env.state[4] < -4.95:
            self.internal_frame = 0
            self.step()
        while True:
            if time.time()-init_time > 0.01:
                return task.cont
    
    def step(self):
        image_in = np.copy(self.images[T_visual_time])
                     
        network_in = torch.Tensor(image_in).to(self.device).detach()
        control_in = torch.Tensor([self.control_network_in]).to(self.device).detach()
        network_in = torch.unsqueeze(network_in, 0)
        visual_action, _ = self.ldg_policy.policy_old(network_in, control_in, torch.zeros([1,3]).to(self.device))
        
        visual_action = visual_action.detach().cpu().numpy().flatten()

        self.vel_error = visual_action*VELOCITY_SCALE+VELOCITY_D
                
        self.reward, self.last_shaping, self.visual_done = visual_reward(TOTAL_STEPS, self.marker_position, self.quad_env.state[0:5:2], self.quad_env.state[1:6:2], self.vel_error, self.last_shaping, self.internal_frame, self.quad_env.ang, self.quad_env.state[-3:])
        self.reward_accum += self.reward
        self.ppo_calls += 1   
        if self.visual_done:
            print('Episode Evaluation: {:.2f}'.format(self.reward_accum), end = '                                                \n')
            self.reward_accum = 0
            self.reset()
