import cv2 as cv
from collections import deque
import time
import numpy as np
import torch
import sys
from collections import deque
from scipy.spatial.transform import Rotation as R
from computer_vision.detector_setup import detection_setup
from environment.quadrotor_env import quad
from environment.controller.model import ActorCritic
from environment.controller.dl_auxiliary import dl_in_gen
from visual_landing.landing_reward_fuction import visual_reward
import matplotlib.pyplot as plt

from visual_landing.ppo_aux import Memory, PPO, evaluate 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

T = 5
TOTAL_TS = 1500
TS = 0.01
update_timestep = 4000
len_image = np.array([160, 160])
PLOT_INTERVAL = 25
class computer_vision():
    def __init__(self, render, quad_model, prop_models, cv_cam, camera_cal, mydir, IMG_POS_DETER, error_display, crtl_display):
        
        self.mtx = camera_cal.mtx
        self.dist = camera_cal.dist
        self.error_display = error_display
        self.crtl_display = crtl_display
        self.IMG_POS_DETER = IMG_POS_DETER
    
        self.quad_env = quad(TS, TOTAL_TS, 1, T)
        self.aux_dl = dl_in_gen(T, self.quad_env.state_size, self.quad_env.action_size)
        state_dim = self.aux_dl.deep_learning_in_size
        self.policy = ActorCritic(state_dim, action_dim=4, action_std=0).to(device)
        try:
            self.policy.load_state_dict(torch.load('./environment/controller/PPO_continuous_solved_drone.pth',map_location=device))
            print('Saved Control policy loaded')
        except:
            print('Could not load Control policy')
            sys.exit(1)           
            
 
        self.render = render  
        
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        self.cv_cam = cv_cam
        self.cv_cam.cam.setPos(0, 0, 0)
        self.cv_cam.cam.setHpr(0, 270, 0)
        self.cv_cam.cam.reparentTo(self.render.quad_model)

        self.render.taskMgr.add(self.ppo_train, 'ppo_training_algorithm')
        
        self.quad_model = quad_model
        self.prop_models = prop_models
        
        self.training_episode = 0
        self.training_timestep = 0
        self.training_last_eval = 0
        self.timestep = 0
        
        state_dim = 800*800
        action_dim = 3
        self.visual_ppo = PPO(state_dim, action_dim)

        self.memory = Memory()
        
        
        
        self.last_shaping = 0
        
        self.velocity_scale = np.array([1, 1, 4])           
        self.vel_error = np.array([0, 0, 0])
        self.old_vel_error = np.array([0, 0, 0])
        self.zero_images()
        self.vel_iterator = 0
        self.visual_done = True
        self.evaluate_flag = False
        self.last_training_episode = 0
        self.image_count = 0
        self.evaluate_reward = []
        self.eval_done = False
        self.fig, self.axs = plt.subplots(2)
        plt.draw()
        plt.pause(1)
        self.action_hist = deque(maxlen=(100))
        self.reward_hist = deque(maxlen=(100))
        self.plot_count = 0
    def quadrotor_env_att(self, vel_error):
        if self.quad_env.done and self.visual_done:
            self.image = np.zeros(len_image)
            self.image_1 = np.zeros(len_image)
            self.image_2 = np.zeros(len_image)
            
            random_marker_position = np.random.normal([0, 0], 0.8)
            self.render.checker.setPos(*tuple(random_marker_position), 0.001)
            self.marker_position = np.append(random_marker_position, 0.001)
           
            states, action = self.quad_env.reset(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
            self.network_in = self.aux_dl.dl_input(states, action)
            pos = self.quad_env.state[0:5:2]
            ang = self.quad_env.ang
            self.a = np.zeros(4)
            done = False
            self.visual_done = False
        else:
            action = self.policy.actor(torch.FloatTensor(self.network_in).to(device)).cpu().detach().numpy()
            states, _, done = self.quad_env.step(action)
            # for i, vel in enumerate(vel_error):
            #     states[0, 1+i*2] = vel         
            vel_state = np.array([0, vel_error[0], 0, vel_error[1], 0, vel_error[2], 0, 0, 0, 0, 0, 0, 0, 0])
            self.network_in = self.aux_dl.dl_input(states-vel_state, [action])
            pos = self.quad_env.state[0:5:2]
            ang = self.quad_env.ang
            for i, w_i in enumerate(self.quad_env.w):
                self.a[i] += (w_i*TS)*180/np.pi/10
        ang_deg = (ang[2]*180/np.pi, ang[0]*180/np.pi, ang[1]*180/np.pi)
        pos = (0+pos[0], 0+pos[1], 5+pos[2])
        
        self.quad_model.setPos(*pos)
        self.quad_model.setHpr(*ang_deg)
        for prop, a in zip(self.prop_models, self.a):
            prop.setHpr(a, 0, 0)
        return done
     
    def zero_images(self):
        self.image = np.zeros(len_image)
        self.image_1 = np.zeros(len_image)
        self.image_2 = np.zeros(len_image)

    
    def ppo_train(self, task):
        if self.image_count >= 3:
            vel_error = np.linspace(self.old_vel_error, self.vel_error, 10)[self.vel_iterator]
        else:
            vel_error = np.linspace(self.old_vel_error, self.vel_error, 36)[self.vel_iterator]

        self.vel_iterator += 1
        done = self.quadrotor_env_att(vel_error)
        
        if self.evaluate_flag:
            if task.frame % self.cv_cam.frame_int == 1: 
                self.image_count += 1
                self.image_2 = self.image_1   
                self.image_1 = self.image                 
                ret, image = self.cv_cam.get_image()  
                self.image = cv.cvtColor(image[:,:,0:4], cv.COLOR_BGR2GRAY)
                if self.image_count >= 3 and ret:
                    ts_reward, _, self.eval_done = visual_reward(self.marker_position, self.quad_env.state[0:5:2], self.quad_env.state[1:6:2], self.vel_error, self.last_shaping)
                    self.evaluate_reward.append(ts_reward)
                    if done:
                        self.eval_done = True
                    self.old_vel_error = self.vel_error    
                    cv.imshow('Drone Image', self.image)
                    cv.waitKey(1)
                    network_in = np.array([self.image/255.0, self.image_1/255.0, self.image_2/255.0])
                    network_in = torch.FloatTensor([network_in]).to(device)
                    visual_action = self.visual_ppo.policy.actor(network_in).cpu().detach().numpy()[0]  
                    self.action_hist.append(visual_action)
                    self.vel_error = visual_action*self.velocity_scale         
                    self.error_display.setText('{:.2f}'.format(ts_reward))
                    self.crtl_display.setText('X {:.2f} Y {:.2f} Z {:.2f}'.format(visual_action[0], visual_action[1], visual_action[2]))
                    
                    
                    self.action_hist.append(visual_action)
                    self.reward_hist.append(ts_reward)
                    
                    if self.plot_count % PLOT_INTERVAL == 0:
                        self.axs[0].clear()
                        self.axs[0].plot(np.arange(len(self.action_hist)),self.action_hist)
                        self.axs[1].clear()
                        self.axs[1].plot(np.arange(len(self.reward_hist)),self.reward_hist)                        
                        self.fig.canvas.draw()
                        self.fig.canvas.flush_events()
                        
                    self.plot_count += 1              
                    
                    
                    
                    
                    self.vel_iterator = 0 
                if self.eval_done:    
                    print('\rEvaluation- Episode n: '+str(self.training_episode)+' reward: {:.2f}'.format(np.sum(self.evaluate_reward)), end='         \n')
                    self.evaluate_reward = []
                    self.old_vel_error = np.zeros(3)
                    self.vel_error = np.zeros(3)
                    self.vel_iterator = 0
                    self.reward = 0
                    self.evaluate_flag = False
                    self.eval_done = False
                    self.visual_done = False
                    self.quad_env.done = True
        else:
            if task.frame % self.cv_cam.frame_int == 1:                 
                self.image_count += 1
                self.image_2 = self.image_1   
                self.image_1 = self.image                 
                ret, image = self.cv_cam.get_image()  

                self.image = cv.cvtColor(image[:,:,0:4], cv.COLOR_BGR2GRAY)
                if self.image_count >= 3 and ret:
                    self.reward, self.last_shaping, self.visual_done = visual_reward(self.marker_position, self.quad_env.state[0:5:2], self.quad_env.state[1:6:2], self.vel_error, self.last_shaping)
                    if done:
                        self.visual_done = True
                    self.old_vel_error = self.vel_error                

                    cv.imshow('Drone Image', cv.hconcat([self.image_2, self.image_1, self.image]))
                    cv.waitKey(1)
                    network_in = np.array([self.image/255.0, self.image_1/255.0, self.image_2/255.0])
    
                    self.memory.rewards.append(self.reward)
                    self.memory.is_terminals.append(self.visual_done)                
                    visual_action = self.visual_ppo.select_action(network_in, self.memory)
                    self.action_hist.append(visual_action)
                    self.reward_hist.append(self.reward)
                    
                    self.crtl_display.setText('X {:.2f} Y {:.2f} Z {:.2f}'.format(visual_action[0], visual_action[1], visual_action[2]))
                    
                    self.timestep += 1
                    
                    if self.plot_count % PLOT_INTERVAL == 0:
                        self.axs[0].clear()
                        self.axs[0].plot(np.arange(len(self.action_hist)),self.action_hist)
                        self.axs[1].clear()
                        self.axs[1].plot(np.arange(len(self.reward_hist)),self.reward_hist)                                            
                        self.fig.canvas.draw()
                        self.fig.canvas.flush_events()

                    self.plot_count += 1
                    
                    self.vel_error = visual_action*self.velocity_scale         
                    self.error_display.setText('{:.2f}'.format(self.reward))
                    self.vel_iterator = 0 
                    print('\rEpisode Progress: {:.2%}          '.format((self.timestep/update_timestep)),end='')
                if self.visual_done: 
                    self.image_count = 0
                    self.old_vel_error = np.zeros(3)
                    self.vel_error = np.zeros(3)
                    self.vel_iterator = 0
                    self.reward = 0
                    self.training_episode += 1
                    if self.timestep > update_timestep:
                            cv.destroyWindow('Drone Image')
                            self.evaluate_flag = True
                            self.training_timestep += 1
                            self.timestep = 0
                            self.visual_ppo.update(self.memory)
                            self.memory.clear_memory()
                            self.last_training_episode = self.training_episode
                            self.zero_images()
                            if self.training_timestep % 10 == 0:
                                print('\r Episode n.{:d} Policy Saved'.format(self.training_episode))
                                torch.save(self.visual_ppo.policy.state_dict(), './PPO_landing.pth')
                                torch.save(self.visual_ppo.policy_old.state_dict(), './PPO_landing_old.pth')
                    self.quad_env.done = True
                    self.last_shaping = None            
        return task.cont
 
