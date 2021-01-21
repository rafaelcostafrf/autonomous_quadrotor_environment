
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
from environment.controller.model import ActorCritic_old
from environment.controller.dl_auxiliary import dl_in_gen
from collections import deque

from visual_landing.ppo_trainer import PPO
from visual_landing.rl_memory import Memory, Memory_2D
from visual_landing.rl_reward_fuction import visual_reward
from visual_landing.memory_leak import debug_gpu
import matplotlib.pyplot as plt

T = 5

conv_3D = False
# conv_3D = True
if conv_3D:
    T_visual_time = [6, 5, 2, 1, 0]
    T_visual = len(T_visual_time)
    T_total = T_visual_time[0]+1
else:
    T_visual_time = 0
    T_visual = 1
    T_total = 1
IMAGE_LEN = np.array([84, 84])
IMAGE_CHANNELS = 3
IMAGE_TIME = T_visual

EVAL_FREQUENCY = 10
EVAL_EPISODES = 20

TIME_STEP = 0.01
TOTAL_STEPS = 1500




TASK_INTERVAL_STEPS = 10
BATCH_SIZE = 256*1
VELOCITY_SCALE = [0.5, 0.5, 1]
VELOCITY_D = [0, 0, -VELOCITY_SCALE[2]/1.5]
#CONTROL POLICY
AUX_DL = dl_in_gen(T, 13, 4)
state_dim = AUX_DL.deep_learning_in_size
CRTL_POLICY = ActorCritic_old(state_dim, action_dim=4, action_std=0)
try:
    CRTL_POLICY.load_state_dict(torch.load('./environment/controller/PPO_continuous_drone_velocity_solved.pth'))
    print('Saved Control policy loaded')
except:
    print('Could not load Control policy')
    sys.exit(1)  

PLOT_LENGTH = 100
CONV_SIZE = 256
       
class quad_worker():
    def __init__(self, render, cv_cam, child_number = None, child = False):
        self.n_solved = 0
        self.update = [0, None]
        self.done = False
        self.render = render
        self.cv_cam = cv_cam
        self.ldg_policy = PPO(3, child, T_visual)
        self.train_time = False
        self.batch_size = BATCH_SIZE
        self.child = child
        self.reward = 0
        if child:
            self.batch_size = int(BATCH_SIZE*0.65)
            self.child_number = child_number
            self.device = torch.device('cpu')  
        else: 
            self.n_samples = 0
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
            # self.device = torch.device('cpu') 
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
        if conv_3D:
            self.memory = Memory(self.batch_size, IMAGE_LEN, IMAGE_TIME, IMAGE_CHANNELS)
        else:
            self.memory = Memory_2D(self.batch_size, IMAGE_LEN, IMAGE_CHANNELS)
        
        #TASK MANAGING
        self.wait_for_task = False
        self.visual_done = False
        self.ppo_calls = 0
        self.render.taskMgr.setupTaskChain('async', numThreads = 16, tickClock = None,
                                   threadPriority = None, frameBudget = None,
                                   frameSync = None, timeslicePriority = None)
        # self.render.taskMgr.add(self.step, 'ppo step', taskChain = 'async')
        self.render.taskMgr.add(self.step, 'ppo step')
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
               
        self.eval_flag = False
        self.reward_accum = 0
        self.train_calls = 0
        
        self.eval_episodes = 0 
        self.eval_reward = []
        self.mean_eval = []

        #PLOT SETUP
        if not self.child:
            self.time_step_plot = deque(maxlen=PLOT_LENGTH)
            self.reward_plot = deque(maxlen=PLOT_LENGTH)
            self.efforts_plot = deque(maxlen=PLOT_LENGTH)
            self.vel_plot = deque(maxlen=PLOT_LENGTH)
            self.done_plot = deque(maxlen=PLOT_LENGTH)
            self.fig, self.axs = plt.subplots(5)
            self.fig.suptitle('Event Viewer')
            plt.draw()
            plt.pause(1)
    
        self.old_conv = torch.zeros([1, CONV_SIZE]).to(self.device)
        
    def quad_reset_random(self):
        random_marker_position = np.random.normal([0, 0], 0.8)
        self.render.checker.setPos(*tuple(random_marker_position), 0.001)
        self.marker_position = np.append(random_marker_position, 0.001)
        
        quad_random_z = -5*np.random.random()+1
        quad_random_xy = self.marker_position[0:2]+(np.random.random(2)-0.5)*abs(-5-quad_random_z)/7*4
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
        if conv_3D:
            self.images = np.zeros([T_visual_time[0]+1, IMAGE_CHANNELS, IMAGE_LEN[0], IMAGE_LEN[0]])
        else:
            self.images = np.zeros([IMAGE_CHANNELS, IMAGE_LEN[0], IMAGE_LEN[0]])
     
    def normalize_hsv(self, image):
        image[0] = image[0]/90-1
        image[1] = image[1]/50-1
        image[2] = image[2]/50-1
        return image      
    
    def image_roll(self, image):

        image_copy = (image[:, :, 0:3]).copy()
        # image_copy = image_.copy()
        self.print_image = image_copy
        
        image_copy = np.swapaxes(image_copy, 2, 0)
        image_copy = np.swapaxes(image_copy, 1, 2)

        
        # image_copy = self.normalize_hsv(image_copy)
        image_copy = image_copy/255.0
        
        self.images = np.roll(self.images, 1, 0)
        if conv_3D:
            self.images[0, :, :, :] = image_copy  
        else:
            self.images = image_copy
            # print(np.shape(self.images))
        
    def image_plot(self):
        total = self.print_image
        # total = self.images[T_visual_time[0]]
        # for t in T_visual_time[1:]:
        #     total = np.hstack((total, self.images[t]))
        # cv.imshow('Imagem',total)
        # cv.waitKey(1)
        
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
        if self.train_time:
            if self.train_calls % EVAL_FREQUENCY == 0:
                self.eval_flag = True
                self.n_solved = 0
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
        self.reset()
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
        torch.save(self.memory.last_conv, child_name+'last_conv.tch')
        torch.save(self.memory.state_value, child_name+'state_value.tch')
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
                    
                    last_conv_temp = torch.load(child_name+ 'last_conv.tch')
                    
                    state_value_temp = torch.load(child_name + 'state_value.tch')
                    
                    self.memory.actions = np.append(self.memory.actions, actions_temp, axis = 0)
                    self.memory.states = np.append(self.memory.states, states_temp, axis = 0)
                    self.memory.logprobs = np.append(self.memory.logprobs, logprobs_temp, axis = 0)
                    self.memory.rewards = np.append(self.memory.rewards, rewards_temp, axis = 0)
                    self.memory.is_terminals = np.append(self.memory.is_terminals, is_terminals_temp, axis = 0) 
                    self.memory.sens = np.append(self.memory.sens, sens_temp, axis = 0)
                    self.memory.last_conv = np.append(self.memory.last_conv, last_conv_temp, axis=0)
                    self.memory.state_value = np.append(self.memory.state_value, state_value_temp, axis=0)
                    del actions_temp
                    del states_temp
                    del logprobs_temp
                    del rewards_temp
                    del is_terminals_temp
                    del sens_temp
                    del last_conv_temp
                    del state_value_temp
                    break
                else:                            
                    time.sleep(0.1)
        self.n_samples += len(self.memory.rewards)
        self.ldg_policy.update(self.memory)   
        self.memory.clear_memory()
        self.reset()
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
        self.cv_cam.cam.setPos(*pos)
        self.render.dlightNP.setPos(*pos)
        for prop, a in zip(self.prop_models, self.a):
            prop.setHpr(a, 0, 0)
        self.render.graphicsEngine.renderFrame()
           
    def step(self, task):
        visual_action = np.zeros(3)
        if self.ppo_calls >=1:
            if conv_3D:
                image_in = self.images[T_visual_time]
            else:
                image_in = self.images
            self.image_plot()
            if self.eval_flag:  
                if conv_3D:
                    network_in = torch.Tensor(np.swapaxes(image_in,1,0)).to(self.device).detach()    
                else:
                    network_in = torch.Tensor(image_in).to(self.device).detach()    
                # network_in = torch.Tensor(np.swapaxes(image_in,1,0)).to(self.device).detach()    
                # network_in = torch.tensor(image_in, dtype=torch.float).to(self.device).detach()    
                control_in = torch.tensor([self.control_network_in], dtype=torch.float).to(self.device).detach()
                network_in = torch.unsqueeze(network_in, 0)
                visual_action = self.ldg_policy.policy_old(network_in, control_in)                
                visual_action = visual_action.detach().cpu().numpy().flatten()
    
            else:
                if conv_3D:
                    network_in = np.swapaxes(image_in, 1, 0)  
                else:
                    network_in = image_in
                # network_in = image_in
                training_percentage = ((self.train_calls%EVAL_FREQUENCY*self.batch_size)+self.memory.index)/(self.batch_size*EVAL_FREQUENCY)
                print('\rTraining Progress: {:.2%}'.format(training_percentage), end='\t')    
                control_in = torch.Tensor([self.control_network_in]).to(self.device).detach()
                visual_action, self.old_conv = self.ldg_policy.select_action(network_in, control_in, self.old_conv, self.memory)
              
            self.vel_error = visual_action*VELOCITY_SCALE+VELOCITY_D

        
        for i in range(TASK_INTERVAL_STEPS):
            self.internal_frame += 1
            states_sens = self.sensor_sp()
            error = np.array([[0, self.vel_error[0], 0, self.vel_error[1], 0, self.vel_error[2], 0, 0, 0, 0, 0, 0, 0, 0]])
            self.control_network_in = self.aux_dl.dl_input(states_sens-error, [self.crtl_action])
            crtl_network_in = torch.tensor(self.control_network_in, dtype=torch.float, device=torch.device('cpu'))
            self.crtl_action = CRTL_POLICY.actor(crtl_network_in).cpu().detach().numpy()
            states, _, _ = self.quad_env.step(self.crtl_action)
            coordinates = np.concatenate((states[0, 0:5:2], self.quad_env.ang, np.zeros(4))) 
            self.render_position(coordinates, self.marker_position)
            image = self.take_picture()
            self.image_roll(image)            
            # if self.quad_env.state[4] < -4.95:
            #     break
        
        self.reward, self.last_shaping, self.visual_done, self.solved = visual_reward(TOTAL_STEPS, self.marker_position, self.quad_env.state[0:5:2], self.quad_env.state[1:6:2], visual_action, self.last_shaping, self.internal_frame, self.quad_env.ang, self.quad_env.state[-3:])            
        if self.ppo_calls >= 1:                   
            if self.eval_flag:
                self.reward_accum += self.reward
            if not self.eval_flag: 
                self.memory.append_memory_rt(self.reward, self.visual_done)
        if self.eval_flag:
            self.n_solved += self.solved        
        
        
        if not self.child :                    
            self.reward_plot.append(self.reward)
            self.time_step_plot.append(self.ppo_calls)
            self.efforts_plot.append(self.vel_error)  
            self.vel_plot.append(self.quad_env.state[1:6:2])
            self.done_plot.append(self.visual_done)
            if self.ppo_calls % 20 == 0:            
                for axis in self.axs[0:2]:
                    axis.clear()
                    axis.grid()     
                self.axs[0].plot(np.arange(len(self.efforts_plot)), np.array(self.efforts_plot)[:, 0], color='g', linewidth=0.5)   
                self.axs[0].plot(np.arange(len(self.efforts_plot)), np.array(self.vel_plot)[:, 0], ls='--', color='g', linewidth=0.5)   
                
                self.axs[0].plot(np.arange(len(self.efforts_plot)), np.array(self.efforts_plot)[:, 1], color='r', linewidth=0.5)    
                self.axs[0].plot(np.arange(len(self.efforts_plot)), np.array(self.vel_plot)[:, 1], ls='--', color='r', linewidth=0.5)   
                
                self.axs[0].plot(np.arange(len(self.efforts_plot)), np.array(self.efforts_plot)[:, 2], color='b', linewidth=0.5)    
                self.axs[0].plot(np.arange(len(self.efforts_plot)), np.array(self.vel_plot)[:, 2], ls='--', color='b', linewidth=0.5)   
                
                self.axs[1].plot(np.arange(len(self.efforts_plot)), np.array(self.reward_plot), color='b', linewidth=0.2)  

                for i, terminal in enumerate(self.done_plot):
                    if terminal:
                        self.axs[0].axvline(x=i, color='m', linewidth=1)
                        self.axs[1].axvline(x=i, color='m', linewidth=1)
                plt.draw()
                self.fig.canvas.flush_events()


        self.ppo_calls += 1   

        if self.memory.index > int(0.85*self.batch_size) and self.visual_done:
            self.train_time = True
            self.memory.close_memory()
        if self.train_time:
            if self.child:
                self.child_save_data()
                self.train_calls += 1
                self.reset_policy()
            else:   
                self.train_calls += 1
                self.mother_train()
                self.axs[2].clear()
                self.axs[2].grid()
                self.axs[2].plot(np.arange(0,self.train_calls), self.ldg_policy.critic_loss_memory)
                self.axs[3].clear()
                self.axs[3].grid()
                self.axs[3].plot(np.arange(0,self.train_calls), self.ldg_policy.actor_loss_memory)
                
        if self.visual_done:
            if self.eval_flag:
                self.eval_episodes += 1
                self.eval_reward.append(self.reward_accum)
                self.reward_accum = 0
                if self.eval_episodes == EVAL_EPISODES:
                    self.eval_episodes = 0
                    self.eval_flag = False
                    self.mean_eval.append(np.mean(self.eval_reward))
                    self.eval_reward = []
                    if not self.child:
                        self.axs[4].clear()
                        self.axs[4].grid()
                        self.axs[4].plot(np.arange(0,len(self.mean_eval)), self.mean_eval)
                        print('Total Steps: {:d}'.format(self.n_samples),end='\t')
                    print('Evaluation Mean: {:.2f} \t Solved: {:.2%}'.format(self.mean_eval[-1], self.n_solved/EVAL_EPISODES))

            self.reset()

        
        # print(torch.cuda.memory_summary(torch.device('cuda:0')))
        return task.cont
