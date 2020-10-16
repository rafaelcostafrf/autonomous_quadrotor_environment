import tracemalloc
tracemalloc.start()


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
T_visual = 3
EVAL_FREQUENCY = 2

TIME_STEP = 0.01
TOTAL_STEPS = 3500

IMAGE_LEN = np.array([128, 128])
TASK_INTERVAL_STEPS = 10
BATCH_SIZE = 250
VELOCITY_SCALE = [1, 1, 2]

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
        quad_random_pos = (np.random.random(3)-0.5)*2
        states, action = self.quad_env.reset(np.array([quad_random_pos[0], 0, quad_random_pos[1], 0, quad_random_pos[2], 0, 1, 0, 0, 0, 0, 0, 0]))
        self.sensor.reset()
        self.aux_dl =  dl_in_gen(T, 13, 4)
        self.control_network_in = self.aux_dl.dl_input(states, action)
        self.image_zeros() 
        self.memory = Memory()
        
        #TASK MANAGING
        self.wait_for_task = False
        self.visual_done = False
        self.ppo_calls = 0
        self.render.taskMgr.setupTaskChain('async', numThreads = 4, tickClock = None,
                                   threadPriority = None, frameBudget = None,
                                   frameSync = None, timeslicePriority = None)
        # self.render.taskMgr.add(self.step, 'ppo step', taskChain = 'async')
        self.render.taskMgr.add(self.step, 'ppo step')
        #LDG TRAINING
        self.vel_error = np.zeros(3)
        self.last_shaping = None
        
        #MARKER POSITION
        random_marker_position = np.random.normal([0, 0], 0.8)
        self.render.checker.setPos(*tuple(random_marker_position), 0.001)
        self.marker_position = np.append(random_marker_position, 0.001)
        self.internal_frame = 0
        
        #CAMERA SETUP
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        self.cv_cam = cv_cam
        self.cv_cam.cam.setPos(0, 0, 0)
        self.cv_cam.cam.setHpr(0, 270, 0)
        self.cv_cam.cam.reparentTo(self.render.quad_model)
               
        self.eval_flag = False
        self.reward_accum = 0
        self.train_calls = 0
        
        #PLOT SETUP
        if not self.child:
            self.time_step_plot = deque(maxlen=100)
            self.reward_plot = deque(maxlen=100)
            self.efforts_plot = deque(maxlen=100)
            self.fig, self.axs = plt.subplots(2)
            self.fig.suptitle('Event Viewer')
            plt.draw()
            plt.pause(1)
    
        
        
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
        self.images = np.zeros([T_visual, IMAGE_LEN[0], IMAGE_LEN[0]])
     
    def image_roll(self, image):
        self.images = np.roll(self.images, 1, axis=0)
        self.images[0, :, :] = image
        
        
    def take_picture(self):
        ret, image = self.cv_cam.get_image() 
        if ret:
            return cv.cvtColor(image[:,:,0:4], cv.COLOR_BGR2GRAY)/255.0
        else:
            return np.zeros([IMAGE_LEN[0], IMAGE_LEN[0]])
   
    def reset(self):
        quad_random_pos = (np.random.random(3)-0.5)*2
        states, action = self.quad_env.reset(np.array([quad_random_pos[0], 0, quad_random_pos[1], 0, quad_random_pos[2], 0, 1, 0, 0, 0, 0, 0, 0]))
        self.sensor.reset()
        self.aux_dl =  dl_in_gen(T, 13, 4)
        self.control_network_in = self.aux_dl.dl_input(states, action)
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
        self.image_zeros()
        if self.train_time :
            if self.train_calls % EVAL_FREQUENCY == 0:
                self.eval_flag = True
                self.reward_accum = 0
            self.train_time = False  
            
    def reset_policy(self):
        while True:
            f = open('./child_data/'+str(self.child_number)+'.txt', 'r')
            try:
                a = int(f.read())
            except:
                a = None
            if a == 2:
                break
            else:
                time.sleep(0.1)            
        self.ldg_policy.policy.load_state_dict(torch.load('./PPO_landing.pth', map_location=self.device))
        self.ldg_policy.policy_old.load_state_dict(torch.load('./PPO_landing_old.pth', map_location=self.device))
                
   
    
    def child_save_data(self):        
        child_name = './child_data/'+str(self.child_number)        
        torch.save(self.memory.actions, child_name+'actions.tch')            
        torch.save(self.memory.states, child_name+'states.tch')  
        torch.save(self.memory.logprobs, child_name+'logprobs.tch')  
        torch.save(self.memory.rewards, child_name+'rewards.tch')  
        torch.save(self.memory.is_terminals, child_name+'is_terminals.tch')  
        torch.save(self.memory.sens, child_name+'sens.tch')
        self.memory.clear_memory()
        f = open(child_name+'.txt','w')
        f.write(str(1))
        f.close()    

    def mother_train(self):
        f = open('./child_data/child_processes.txt', 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            while True:
                s = open('./child_data/'+line.splitlines()[0]+'.txt', 'r')                            
                try:
                    a = int(s.read())
                except :
                    a = None
                s.close()
                if a == 1:
                    child_name = './child_data/'+line.splitlines()[0]
                   
                    actions_temp = torch.load(child_name+'actions.tch')
                        
                    states_temp = torch.load(child_name+'states.tch')
                        
                    logprobs_temp = torch.load(child_name+'logprobs.tch')
                     
                    rewards_temp = torch.load(child_name+'rewards.tch')
                    
                    is_terminals_temp = torch.load(child_name+'is_terminals.tch')
                    
                    sens_temp = torch.load(child_name+'sens.tch')
                    self.memory.actions = np.append(self.memory.actions, actions_temp, axis = 0)
                    self.memory.states = np.append(self.memory.states, states_temp, axis = 0)
                    self.memory.logprobs = np.append(self.memory.logprobs, logprobs_temp, axis = 0)
                    self.memory.rewards = np.append(self.memory.rewards, rewards_temp, axis = 0)
                    self.memory.is_terminals = np.append(self.memory.is_terminals, is_terminals_temp, axis = 0) 
                    self.memory.sens = np.append(self.memory.sens, sens_temp, axis = 0)
                    del actions_temp
                    del states_temp
                    del logprobs_temp
                    del rewards_temp
                    del is_terminals_temp
                    del sens_temp
                    break
                else:                            
                    time.sleep(0.1)
        self.n_samples += len(self.memory.rewards)
        print('Total Number of Samples: {:d}'.format(self.n_samples), end='\t')           
        self.ldg_policy.update(self.memory)   
        self.memory.clear_memory()

        gc.collect()

        
        for line in lines:
            s = open('./child_data/'+line.splitlines()[0]+'.txt', 'w')    
            s.write(str(2))
            s.close()


        
        
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
           
    def step(self, task):
        for i in range(TASK_INTERVAL_STEPS):

            self.internal_frame += 1
            # LOWER CONTROL STEP  
            crtl_network_in = torch.FloatTensor(self.control_network_in).to('cpu')

            crtl_action = CRTL_POLICY.actor(crtl_network_in).cpu().detach().numpy()

            states, _, _ = self.quad_env.step(crtl_action)
            states_sens = self.sensor_sp()
            
            # CONTROL DIFFERENCE
            error_smoothing = states_sens
            error_smoothing[0, 1] -= self.vel_error[0] #- error_smoothing[0, 1]
            error_smoothing[0, 3] -= self.vel_error[1] #- error_smoothing[0, 3]
            error_smoothing[0, 5] -= self.vel_error[2] #- error_smoothing[0, 5]
            
            # print(error_smoothing, self.vel_error)
            self.control_network_in = self.aux_dl.dl_input(error_smoothing, [crtl_action])
                    
            coordinates = np.concatenate((states[0, 0:5:2], self.quad_env.ang, np.zeros(4))) 


        self.reward, self.last_shaping, self.visual_done = visual_reward(TOTAL_STEPS, self.marker_position, self.quad_env.state[0:5:2], self.quad_env.state[1:6:2], self.vel_error, self.last_shaping, self.internal_frame)
        
        self.render_position(coordinates, self.marker_position)
        # self.render.graphicsEngine.renderFrame()
        image = self.take_picture()
        self.image_roll(image)
                
        # if self.child:
        #     if self.child_number == 0 :
        #         for i, image in enumerate(self.images):
        #             if i == 0:
        #                 a = image
        #             else:
        #                 a = np.hstack((a, image))
        #         cv.imshow('teste',a)
        #         cv.waitKey(1)
                
        
        if self.ppo_calls >= T_visual:           
            if self.eval_flag:
                network_in = torch.Tensor(self.images).to(self.device).detach()
                control_in = torch.Tensor([self.control_network_in]).to(self.device).detach()
                network_in = torch.unsqueeze(network_in, 0)
                visual_action, _ = self.ldg_policy.policy_old(network_in, control_in)
                
                visual_action = visual_action.detach().cpu().numpy().flatten()

                self.reward_accum += self.reward
            else:
                print('\rBatch Progress: {:.2%}'.format(len(self.memory.states)/BATCH_SIZE), end='          ')
                if len(self.memory.states) == BATCH_SIZE:
                    self.visual_done = True
                    self.train_time = True
                
                self.memory.rewards = np.append(self.memory.rewards, self.reward)
            
                self.memory.is_terminals = np.append(self.memory.is_terminals, self.visual_done)   
                                
                network_in = self.images
                control_in = torch.Tensor([self.control_network_in]).to(self.device).detach()
                visual_action = self.ldg_policy.select_action(network_in, control_in, self.memory)
            
            self.vel_error = visual_action*VELOCITY_SCALE    

        if not self.child :                    
            self.reward_plot.append(self.reward)
            self.time_step_plot.append(self.ppo_calls)
            self.efforts_plot.append(self.vel_error)        
            if self.ppo_calls % 20 == 0:            
                self.axs[0].clear()            
                self.axs[1].clear()            
                self.axs[0].plot(np.arange(len(self.efforts_plot)), self.efforts_plot)    
                self.axs[1].plot(np.arange(len(self.efforts_plot)), self.reward_plot)
                plt.draw()
                self.fig.canvas.flush_events()


        self.ppo_calls += 1   
        # print(self.ppo_calls)
        if self.train_time:
            if self.child:
                # print('Memory Usage: {:.2f}Mb'.format(process.memory_info().rss/1000000), end='\t')
                self.child_save_data()
                self.train_calls += 1
                self.reset_policy()
            else:   
                # print('Before Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000), end='\t')
                self.mother_train() 
                self.train_calls += 1
                # print('After Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000))

                
        if self.visual_done:
            if self.eval_flag:
                self.eval_flag = False
                print('Episode Evaluation: {:.2f}'.format(self.reward_accum), end = '                                                \n')
                self.reward_accum = 0
            # print('Step Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
            self.reset()

        
        # print(torch.cuda.memory_summary(torch.device('cuda:0')))
        return task.cont
