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
from environment.quadrotor_env_opt import quad
from environment.controller.model import ActorCritic
from environment.controller.dl_auxiliary import dl_in_gen


from visual_landing.ppo_aux_v2 import PPO
from visual_landing.v2.memory import Memory
from visual_landing.v2.landing_reward_fuction import visual_reward
from visual_landing.memory_leak import debug_gpu


T = 5
TIME_STEP = 0.01
TOTAL_STEPS = 1500
IMAGE_LEN = np.array([160, 160])
TASK_INTERVAL_STEPS = 10
BATCH_SIZE = 300
VELOCITY_SCALE = [1, 1, 4]

#CONTROL POLICY
AUX_DL = dl_in_gen(T, 13, 4)
state_dim = AUX_DL.deep_learning_in_size
CRTL_POLICY = ActorCritic(state_dim, action_dim=4, action_std=0).to(torch.device('cpu'))
try:
    CRTL_POLICY.load_state_dict(torch.load('./environment/controller/PPO_continuous_solved_drone.pth',map_location=torch.device('cpu')))
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
        self.ldg_policy = PPO(3, 3)
        self.train_time = False

        self.child = child
        if child:
            self.child_number = child_number
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        else:
            self.device = torch.device('cpu')
                  
        
        self.quad_model = render.quad_model
        self.prop_models = render.prop_models
        self.a = np.zeros(4)
        
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
        self.render.taskMgr.setupTaskChain('async', numThreads = 2, tickClock = None,
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
        

    def image_zeros(self):
        self.image_1 = np.zeros(IMAGE_LEN) 
        self.image_2 = np.zeros(IMAGE_LEN)
        self.image_3 = np.zeros(IMAGE_LEN)   
     
    def image_roll(self, image):
        self.image_3 = self.image_2
        self.image_2 = self.image_1
        self.image_1 = image
        
    def take_picture(self):
        ret, image = self.cv_cam.get_image() 
        if ret:
            return cv.cvtColor(image[:,:,0:4], cv.COLOR_BGR2GRAY)/255.0
        else:
            return np.zeros([160, 160])
   
    def reset(self):
        states, action = self.quad_env.reset(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
        self.aux_dl =  dl_in_gen(T, 13, 4)
        self.network_in = self.aux_dl.dl_input(states, action)
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
    
    def reset_policy(self):
        while True:
            f = open('./child_data/'+str(self.child_number)+'.txt', 'r')
            if int(f.read()) == 0:
                break
            else:
                time.sleep(3)            
        self.ldg_policy.policy.load_state_dict(torch.load('./PPO_landing.pth', map_location=self.device))
        self.ldg_policy.policy_old.load_state_dict(torch.load('./PPO_landing_old.pth', map_location=self.device))
                
   
    
    def child_save_data(self):        
        child_name = './child_data/'+str(self.child_number)        
        torch.save(self.memory.actions, child_name+'actions.tch')            
        torch.save(self.memory.states, child_name+'states.tch')  
        torch.save(self.memory.logprobs, child_name+'logprobs.tch')  
        torch.save(self.memory.rewards, child_name+'rewards.tch')  
        torch.save(self.memory.is_terminals, child_name+'is_terminals.tch')  
        self.memory.clear_memory()
        f = open(child_name+'.txt','w')
        f.write(str(1))
        f.close()    

    def mother_train(self):
        f = open('child_processes.txt', 'r')
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
                    
                    self.memory.actions = np.append(self.memory.actions, actions_temp, axis = 0)
                    self.memory.states = np.append(self.memory.states, states_temp, axis = 0)
                    self.memory.logprobs = np.append(self.memory.logprobs, logprobs_temp, axis = 0)
                    self.memory.rewards = np.append(self.memory.rewards, rewards_temp, axis = 0)
                    self.memory.is_terminals = np.append(self.memory.is_terminals, is_terminals_temp, axis = 0)             
                    del actions_temp
                    del states_temp
                    del logprobs_temp
                    del rewards_temp
                    del is_terminals_temp
                    break
                else:                            
                    time.sleep(3)
                    
        self.ldg_policy.update(self.memory)   
        self.memory.clear_memory()

        gc.collect()

        
        for line in lines:
            s = open('./child_data/'+line.splitlines()[0]+'.txt', 'w')    
            s.write(str(0))
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

        for prop, a in zip(self.prop_models, self.a):
            prop.setHpr(a, 0, 0)
         
    def eval_episode(self):
        self.reset()
        done = False
        reward = 0
        while not done:
            with torch.no_grad():
                for i in range(TASK_INTERVAL_STEPS):
        
                    self.internal_frame += 1
                    # LOWER CONTROL STEP  
                    crtl_network_in = torch.Tensor(self.network_in).to(self.device)
        
                    crtl_action = CRTL_POLICY.actor(crtl_network_in).cpu().detach().numpy()
        
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
        
        
                instant_reward, _, done = visual_reward(self.marker_position, self.quad_env.state[0:5:2], self.quad_env.state[1:6:2], self.vel_error, self.last_shaping, self.internal_frame)
                reward += instant_reward
                
                self.render_position(coordinates, self.marker_position)
                
                image = self.take_picture()
                
                # image_concat = np.hstack((self.image_1, self.image_2, self.image_3))
                # cv.imshow('teste', image_concat)
                # cv.waitKey(1)
                
                self.image_roll(image)
            
                
                if self.ppo_calls >= 3:
                    
                    network_in = np.array([self.image_1, self.image_2, self.image_3])
                    network_in = torch.Tensor([network_in]).to(self.device)
                    
                    visual_action = self.ldg_policy.policy.actor(network_in).cpu().detach().numpy().flatten()
        
                    self.vel_error = visual_action*VELOCITY_SCALE            
                
                self.ppo_calls += 1   
        print('Episode Evaluation: {:.2f}'.format(reward), end = '\n')
        self.reset()
    
    def step(self, task):

        with torch.no_grad():
            for i in range(TASK_INTERVAL_STEPS):
    
                self.internal_frame += 1
                # LOWER CONTROL STEP  
                crtl_network_in = torch.FloatTensor(self.network_in).to(self.device)
    
                crtl_action = CRTL_POLICY.actor(crtl_network_in).cpu().detach().numpy()
    
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
    
    
            self.reward, self.last_shaping, self.visual_done = visual_reward(self.marker_position, self.quad_env.state[0:5:2], self.quad_env.state[1:6:2], self.vel_error, self.last_shaping, self.internal_frame)
            
            self.render_position(coordinates, self.marker_position)
            self.render.graphicsEngine.renderFrame()
            image = self.take_picture()
            image_concat = np.hstack((self.image_1, self.image_2, self.image_3))
            cv.imshow('teste', image_concat)
            cv.waitKey(1)
            self.image_roll(image)
        
            
            if self.ppo_calls >= 3:
                
                if len(self.memory.states) == BATCH_SIZE:
                    self.visual_done = True
                    self.train_time = True
                
                self.memory.rewards = np.append(self.memory.rewards, self.reward)
            
                self.memory.is_terminals = np.append(self.memory.is_terminals, self.visual_done)   
                
                network_in = np.array([self.image_1, self.image_2, self.image_3])
               
                
                visual_action = self.ldg_policy.select_action(network_in, self.memory)
                
    
                self.vel_error = visual_action*VELOCITY_SCALE    
                
            self.ppo_calls += 1   

        if self.train_time:
            if self.child:
                print('Memory Usage: {:.2f}Mb'.format(process.memory_info().rss/1000000))
                self.child_save_data()
                self.reset_policy()
            else:   
                print('Before Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000))
                self.mother_train() 
                print('After Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000), end='          ')
                self.eval_episode()
            self.train_time = False  

                
        if self.visual_done:
            # print('Step Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000))   
            self.reset()

        print('\rBatch Progress: {:.2%}'.format(len(self.memory.states)/BATCH_SIZE), end='          ')

        return task.cont