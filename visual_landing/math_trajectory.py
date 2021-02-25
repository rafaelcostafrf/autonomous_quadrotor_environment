import sys
sys.path.append('/home/rafaelcostaf/mestrado/quadrotor_environment/')


import torch
import numpy as np 
from environment.quadrotor_env import quad, sensor
from environment.controller.model import ActorCritic_old
from environment.controller.dl_auxiliary import dl_in_gen
from visual_landing.rl_reward_fuction import visual_reward
from environment.quaternion_euler_utility import deriv_quat

device = torch.device('cpu')

T = 5
N = 64
#CONTROL POLICY
AUX_DL = dl_in_gen(T, 13, 4)
state_dim = AUX_DL.deep_learning_in_size
CRTL_POLICY = ActorCritic_old(N, state_dim, action_dim=4, action_std=0.0)
try:
    CRTL_POLICY.load_state_dict(torch.load('/home/rafaelcostaf/mestrado/quadrotor_environment/visual_landing/controller/PPO_continuous_drone_velocity_solved.pth'))
    print('Saved Control policy loaded')
except:
    print('Could not load Control policy')
    sys.exit(1)  


time_int_step = 0.01
max_timesteps = 3000
quad_env = quad(time_int_step, max_timesteps, training = False, euler=0, direct_control=1, T=T)
sensor = sensor(quad_env)  

def quad_reset_random():
    random_marker_position = np.random.normal([0, 0], 0.8)
    marker_position = np.append(random_marker_position, 0.001)
    
    quad_random_z = -5*np.random.random()+1
    quad_random_xy = marker_position[0:2]+(np.random.random(2)-0.5)*abs(-5-quad_random_z)/7*4
    initial_state = np.array([quad_random_xy[0], 0, quad_random_xy[1], 0, quad_random_z, 0, 1, 0, 0, 0, 0, 0, 0])

    states, action = quad_env.reset(initial_state)
    return states, action, marker_position


class pos_pi():
    def __init__(self, P, D, I, dt):
        self.P = P
        self.I = I
        self.D = D
        self.dt = dt
        self.integrator = np.zeros(3)
        
    def vel_error(self, s, s_d, ds, ds_d):
        self.integrator += (s_d - s)*self.dt
        error = self.P*(s_d - s) + self.D*(ds_d - ds) + self.I*(self.integrator)
        return error
   
    def reset(self):
        self.integrator = np.zeros(3)

def sensor_sp(sensor):               
            _, velocity_accel, pos_accel = sensor.accel_int()
            quaternion_gyro = sensor.gyro_int()
            ang_vel = sensor.gyro()
            quaternion_vel = deriv_quat(ang_vel, quaternion_gyro)
            pos_gps, vel_gps = sensor.gps()
            # print(pos_gps, vel_gps)
            # print(pos_accel, velocity_accel)
            quaternion_triad, _ = sensor.triad()
            if GPS:
                pos = ((100-GPS_P)*pos_accel + GPS_P*pos_gps)/100
                # print(pos)
                vel = ((100-GPS_P)*velocity_accel + GPS_P*vel_gps)/100
                # print(vel)
                sensor.position_t0 = pos
                sensor.velocity_t0 = vel               
            else:
                pos = pos_accel
                vel = velocity_accel
            pos_vel = np.array([pos[0], vel[0],
                                pos[1], vel[1],
                                pos[2], vel[2]])
            states_sens = np.array([np.concatenate((pos_vel, quaternion_gyro, quaternion_vel))   ])
            return states_sens


VELOCITY_SCALE = np.array([0.5, 0.5, 1])
VELOCITY_D = np.array([0, 0, -VELOCITY_SCALE[2]/1.5])

EP = 500

delta_v = 0
solved = 0
time = np.zeros(EP)

ERROR_POS = np.zeros([EP, max_timesteps, 3])    
VEL = np.zeros([EP, max_timesteps, 3])    
CONTROL = np.zeros([EP, max_timesteps, 3])   

pos_control = pos_pi(4.5, 0.5, 0.00, time_int_step)

MEMS = False
GPS = False
GPS_P = 0.1
for i in range(EP):
    AUX_DL.reset()
    state, action, marker_position = quad_reset_random()
    network_in = AUX_DL.dl_input(state, action)
    last_shaping = None
    sensor.reset()
    # print(state[0])
    for k in range(max_timesteps):
        
        action = CRTL_POLICY.actor(torch.FloatTensor(network_in).to(device)).cpu().detach().numpy()  
        state, reward, done = quad_env.step(action)
        
        if MEMS:
            state = sensor_sp(sensor)
            
        s = state[0, 0:5:2]
        ds = state[0, 1:6:2]
        s_d = np.array([marker_position[0], marker_position[1], -5])
        ds_d = np.zeros(3)
    
        vel_error = pos_control.vel_error(s, s_d, ds, ds_d)
        vel_error = np.clip(vel_error, np.array([-0.5, -0.5, -1.666]), np.array([0.5, 0.5, 0.333]))
                    
        control = (vel_error - VELOCITY_D)/VELOCITY_SCALE
        

        state_error = np.zeros([1, 14])
        state_error[0, 1] = vel_error[0]
        state_error[0, 3] = vel_error[1]
        state_error[0, 5] = vel_error[2]
        
        network_in = AUX_DL.dl_input(state-state_error, [action])
        
        quad_position = quad_env.state[0:5:2]
        quad_vel = quad_env.state[1:6:2]
        ang = quad_env.ang
        v_ang = quad_env.state[-3:]
        # print(quad_position)
        
        reward, last_shaping, done_visual, n_solved = visual_reward(k, marker_position, quad_position, quad_vel, control, last_shaping, k, ang, v_ang)
        
        ERROR_POS[i, k, :] = quad_env.state[0:5:2] - marker_position + np.array([0, 0, 5])
        VEL[i, k, :] = quad_env.state[1:6:2]
        CONTROL[i, k, :] = vel_error
        delta_v += np.sum(np.abs(quad_env.state[1:6:2]))
        
        if done_visual:
            solved += n_solved
            time[i] = k*time_int_step
            print('\rProgress: {:.2%}'.format(i/EP), end='')
            break
print('\rSolved: {:.2%} Avg Time: {:.2f} delta_v: {:.2f}'.format(solved/EP, np.mean(time), delta_v/EP))