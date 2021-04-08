
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
from environment.quadrotor_env import quad, sensor
from environment.quaternion_euler_utility import deriv_quat
from environment.controller.model import ActorCritic_old, ActorCritic
from environment.controller.dl_auxiliary import dl_in_gen
from collections import deque

from visual_landing.ppo_trainer import PPO
from visual_landing.rl_memory import Memory_2D
from visual_landing.rl_reward_fuction import visual_reward
from visual_landing.memory_leak import debug_gpu
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

N = 64
plot_eval = True
save_pgf = True
nome = './resultados/pouso_autonomo/pouso_aleatorio2'
if save_pgf:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.preamble':[
            '\DeclareUnicodeCharacter{2212}{-}']
    })


EVAL_TOTAL = 100

T = 5
T_visual_time = 0
T_visual = 1
T_total = 1
EVAL_FREQUENCY = 1

TIME_STEP = 0.01
TOTAL_STEPS = 1500

IMAGE_LEN = np.array([84, 84])
IMAGE_CHANNELS = 3
TASK_INTERVAL_STEPS = 10
BATCH_SIZE = 356
VELOCITY_SCALE = [0.5, 0.5, 1]
VELOCITY_D = [0, 0, -VELOCITY_SCALE[2]/1.5]
#CONTROL POLICY
AUX_DL = dl_in_gen(T, 13, 4)
state_dim = AUX_DL.deep_learning_in_size
CRTL_POLICY = ActorCritic_old(N, state_dim, action_dim=4, action_std=0)
# try:
CRTL_POLICY.load_state_dict(torch.load('./visual_landing/controller/PPO_continuous_drone_velocity_solved.pth'))
print('Saved Control policy loaded')
# except:
#     print('Could not load Control policy')
#     sys.exit(1)  

PLOT_LENGTH = 100
CONV_SIZE = 256

ERROR_POS = np.zeros([EVAL_TOTAL, TOTAL_STEPS, 3])    
VEL = np.zeros([EVAL_TOTAL, TOTAL_STEPS,3])    
CONTROL = np.zeros([EVAL_TOTAL, TOTAL_STEPS,3])    

class quad_worker():
    def __init__(self, render, cv_cam, child_number = None, child = False):
        print(time.time())
        self.n_episodes = 0
        self.total_solved = 0
        self.total_reward = 0
        self.total_time = 0
        self.delta_v = 0
        self.update = [0, None]
        self.done = False
        self.render = render
        self.cv_cam = cv_cam
        self.ldg_policy = PPO(3, child, T_visual)
        self.train_time = False
        self.batch_size = 1024
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
        self.memory = Memory_2D(self.batch_size, IMAGE_LEN, IMAGE_CHANNELS)
        
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
        # self.cv_cam.cam.reparentTo(self.render.quad_model)
               
        self.eval_flag = True
        self.reward_accum = 0
        self.train_calls = 0
           
        self.old_conv = torch.zeros([1, CONV_SIZE]).to(self.device)
        
    def quad_reset_random(self):
        random_marker_position = np.random.normal([0, 0], 0.8)
        self.render.checker.setPos(*tuple(random_marker_position), 0.001)
        self.marker_position = np.append(random_marker_position, 0.001)
        
        quad_random_z = -5*np.random.random()+1
        quad_random_xy = self.marker_position[0:2]+(np.random.random(2)-0.5)*abs(-5-quad_random_z)/7*4
        initial_state = np.array([quad_random_xy[0], 0, quad_random_xy[1], 0, quad_random_z, 0, 1, 0, 0, 0, 0, 0, 0])
        states, action = self.quad_env.reset(initial_state)
        
        
        distancia = np.array([quad_random_z+5, quad_random_xy[0] - random_marker_position[0], quad_random_xy[1] - random_marker_position[1]])
        distancia = np.linalg.norm(distancia)
        if plot_eval:
            print('Initial Distance: {:.2f}'.format(distancia))
        
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
        print('----------------------------------------------')
        print(states_sens)
        print(self.quad_env.state.flatten())
        return states_sens
            
    def image_zeros(self):
        self.images = np.zeros([IMAGE_CHANNELS, IMAGE_LEN[0], IMAGE_LEN[0]])
     
    def image_roll(self, image):

        image_copy = (image[:, :, 0:3]).copy()
        # image_copy = image_.copy()
        self.print_image = image_copy
        
        image_copy = np.swapaxes(image_copy, 2, 0)
        image_copy = np.swapaxes(image_copy, 1, 2)

        
        # image_copy = self.normalize_hsv(image_copy)
        image_copy = image_copy/255.0
        
        self.images = image_copy

    def take_picture(self):
        ret, image = self.cv_cam.get_image() 
        if ret:
            # out =  cv.cvtColor(image, cv.COLOR_BGR2GRAY)/255.0
              # out =  cv.cvtColor(image, cv.COLOR_BGR2HSV)
              out = image
        else:
            out = np.zeros([IMAGE_LEN, IMAGE_LEN, 4])
        return out
   
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
        self.cv_cam.cam.setPos(*pos)
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
        image = self.take_picture()
        self.image_roll(image)
        self.delta_v += np.sum(np.abs(states[0, 1:6:2]))
        
        if plot_eval:
            ERROR_POS[self.n_episodes-1, self.quad_env.i, :] = states[0, 0:5:2] - self.marker_position + np.array([0, 0, 5])
            VEL[self.n_episodes-1, self.quad_env.i, :] = states[0, 1:6:2]
            CONTROL[self.n_episodes-1, self.quad_env.i, :] = self.vel_error
                
        if self.internal_frame == TASK_INTERVAL_STEPS:
            self.internal_frame = 0
            self.step()
            
        while True:
            if self.n_episodes >= EVAL_TOTAL:
                return task.done
            
            if time.time()-init_time > 0.01:
                return task.cont
            
    
    def step(self):
        network_in = torch.Tensor(self.images).to(self.device).detach()    
        network_in = torch.unsqueeze(network_in, 0)
        
        control_in = torch.Tensor([self.control_network_in]).to(self.device).detach()
        visual_action = self.ldg_policy.policy_old(network_in, control_in)
        
        visual_action = visual_action.detach().cpu().numpy().flatten()

        self.vel_error = visual_action*VELOCITY_SCALE+VELOCITY_D
                
        self.reward, self.last_shaping, self.visual_done, self.solved = visual_reward(TOTAL_STEPS, self.marker_position, self.quad_env.state[0:5:2], self.quad_env.state[1:6:2], self.vel_error, self.last_shaping, self.internal_frame, self.quad_env.ang, self.quad_env.state[-3:])
        self.reward_accum += self.reward
        self.ppo_calls += 1   
        if self.visual_done:
            if plot_eval:
                P = 0.7
                ERROR_AVG = np.mean(ERROR_POS, axis=0)
                VEL_AVG = np.mean(VEL, axis=0)
                CONTROL_AVG = np.mean(CONTROL, axis=0)
                
                ERROR_STD = np.std(ERROR_POS, axis=0)
                VEL_STD = np.std(VEL, axis=0)
                CONTROL_STD = np.std(CONTROL, axis=0)
                plot_time = np.arange(0, TOTAL_STEPS*TIME_STEP, TIME_STEP)
                
                
                
                labels = ['$E_X$ (m)', '$E_Y$ (m)', '$E_Z$ (m)']

                fig, axs = plt.subplots(4, 1, figsize = (7, 7*1.414))
                
                lines = axs[0].plot(plot_time[10:self.quad_env.i], ERROR_AVG[10:self.quad_env.i, 0:3])
                
                lines_1 = axs[1].plot(plot_time[10:self.quad_env.i], VEL_AVG[10:self.quad_env.i, 0], color = 'r', label = '$\dot X$ (m/s)')
                lines_2 = axs[1].plot(plot_time[10:self.quad_env.i], CONTROL_AVG[10:self.quad_env.i, 0], ls = '--', color = 'r', label = '$\dot{X}_d$ (m/s)')
                
                lines_3 = axs[2].plot(plot_time[10:self.quad_env.i], VEL_AVG[10:self.quad_env.i, 1], color = 'b', label = '$\dot Y$ (m/s)')
                lines_4 = axs[2].plot(plot_time[10:self.quad_env.i], CONTROL_AVG[10:self.quad_env.i, 1], ls = '--', color = 'b', label = '$\dot{Y}_d$ (m/s)')
                
                lines_5 = axs[3].plot(plot_time[10:self.quad_env.i], VEL_AVG[10:self.quad_env.i, 2], color = 'g', label = '$\dot Z$ (m/s)')
                lines_6 = axs[3].plot(plot_time[10:self.quad_env.i], CONTROL_AVG[10:self.quad_env.i, 2], ls = '--', color = 'g', label = '$\dot{Z}_d$ (m/s)')
                
                axs[3].set_xlabel('Tempo (s)')
                
                axs[0].legend(lines, labels)
                axs[1].legend()
                axs[2].legend()
                axs[3].legend()
    
    
                for axis in axs:
                    axis.grid(True)
                
                if save_pgf:
                    plt.savefig(nome+'.pgf', bbox_inches='tight')     
                else:
                    plt.show()
                
            self.n_episodes += 1
            self.total_reward += self.reward_accum
            self.total_solved += self.solved
            self.total_time +=  self.quad_env.i*TIME_STEP
            print('Episode Evaluation: {:.2f} \tAverage Reward: {:.2f} \tAverage Solved: {:.2%} \tEpisodes: {:d} \tTime: {:.2f} \tDelta V: {:.2f}'.format(self.reward_accum, self.total_reward/self.n_episodes, self.total_solved/self.n_episodes, self.n_episodes, self.total_time/self.n_episodes, self.delta_v/self.n_episodes), end = '                                                \n')
            self.reward_accum = 0
            self.reset()
