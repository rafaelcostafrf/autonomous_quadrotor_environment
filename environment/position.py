import torch
import time
import numpy as np
import sys
from environment.quadrotor_env import quad, sensor
from environment.quaternion_euler_utility import deriv_quat
from environment.controller.model import ActorCritic
from environment.controller.dl_auxiliary import dl_in_gen

## PPO SETUP ##
time_int_step = 0.01
max_timesteps = 3000
T = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class quad_position():
    
    def __init__(self, render, quad_model, prop_models, REAL_CTRL):
        self.REAL_CTRL = REAL_CTRL
        self.quad_model = quad_model
        self.prop_models = prop_models

        self.time_total_sens = []
        self.T = T
        self.render = render
        self.render.taskMgr.add(self.drone_position_task, 'Drone Position')
        # ENV SETUP
        self.env = quad(time_int_step, max_timesteps, direct_control=1, T=T)
        self.sensor = sensor(self.env)
        self.aux_dl = dl_in_gen(T, self.env.state_size, self.env.action_size)    
        self.error = []
        state_dim = self.aux_dl.deep_learning_in_size
        self.policy = ActorCritic(state_dim, action_dim=4, action_std=0).to(device)
        #CONTROLLER SETUP
        try:
            self.policy.load_state_dict(torch.load('./environment/controller/PPO_continuous_solved_drone.pth',map_location=device))
            print('Saved policy loaded')
        except:
            print('Could not load policy')
            sys.exit(1)
            
        n_parameters = sum(p.numel() for p in self.policy.parameters())
        print('Neural Network Number of Parameters: %i' %n_parameters)

        
    def drone_position_task(self, task):
        if task.frame == 0 or self.env.done:
            # in_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            in_state = None
            states, action = self.env.reset(in_state)
            self.network_in = self.aux_dl.dl_input(states, action)
            self.sensor.reset()
            pos = self.env.state[0:5:2]
            ang = self.env.ang
            self.a = np.zeros(4)
        else:
            action = self.policy.actor(torch.FloatTensor(self.network_in).to(device)).cpu().detach().numpy()
            states, _, done = self.env.step(action)
            time_iter = time.time()
            _, self.velocity_accel, self.pos_accel = self.sensor.accel_int()
            self.quaternion_gyro = self.sensor.gyro_int()
            self.ang_vel = self.sensor.gyro()
            quaternion_vel = deriv_quat(self.ang_vel, self.quaternion_gyro)
            self.pos_gps, self.vel_gps = self.sensor.gps()
            self.quaternion_triad, _ = self.sensor.triad()
            self.time_total_sens.append(time.time() - time_iter)
            
            #SENSOR CONTROL
            pos_vel = np.array([self.pos_accel[0], self.velocity_accel[0],
                                self.pos_accel[1], self.velocity_accel[1],
                                self.pos_accel[2], self.velocity_accel[2]])
            if self.REAL_CTRL:
                self.network_in = self.aux_dl.dl_input(states, [action])
            else:
                states_sens = [np.concatenate((pos_vel, self.quaternion_gyro, quaternion_vel))]                
                self.network_in = self.aux_dl.dl_input(states_sens, [action])
            
            self.error = np.concatenate((-self.env.state[0:6], np.array([1, 0, 0, 0])-self.env.state[6:10], -self.env.state[-3:]))
            # if len(self.error) > 1e4:
            #     error_final = np.array(self.error)
            #     np.savez('./data/hover_mems_results.npz',error_final)
            #     sys.exit()
            pos = self.env.state[0:5:2]
            ang = self.env.ang
            for i, w_i in enumerate(self.env.w):
                self.a[i] += (w_i*time_int_step )*180/np.pi/10
    
        ang_deg = (ang[2]*180/np.pi, ang[0]*180/np.pi, ang[1]*180/np.pi)
        pos = (0+pos[0], 0+pos[1], 5+pos[2])
        
        # self.quad_model.setHpr((0.8, -0.3, -7.7))
        # self.quad_model.setPos((0.595, -0.729, 5.091))
        self.quad_model.setPos(*pos)
        self.quad_model.setHpr(*ang_deg)
        for prop, a in zip(self.prop_models, self.a):
            prop.setHpr(a, 0, 0)
        return task.cont