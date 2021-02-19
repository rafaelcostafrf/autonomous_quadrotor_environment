import sys
sys.path.append('/home/rafaelcostaf/mestrado/quadrotor_environment/')

import numpy as np 
from scipy.linalg import solve_continuous_are as solve_lqr
from environment.quadrotor_env import quad, plotter

import pandas as pd 


n_episodes = 500
max_timesteps = 1500
memory_array = np.zeros([n_episodes, max_timesteps, 13])

solved = 0 

I_xx = 16.83e-3
I_yy = 16.83e-3
I_zz = 28.34e-3
M = 1.03
G = 9.82

clipped = False

if clipped:
    Q_att = np.array([[5, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 5, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0.05, 0],
                      [0, 0, 0, 0, 0, 0.01]])*50
    
    R_att = np.diag(np.ones(4))*40
    
    
    Q_t = np.array([[1e-08, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1e-08, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1e-08, 0],
                    [0, 0, 0, 0, 0, 0.8]])*10
    
    R_t = np.diag(np.ones(3))*10
else:
    Q_att = np.array([[5, 0, 0, 0, 0, 0],
                      [0, 0.3, 0, 0, 0, 0],
                      [0, 0, 5, 0, 0, 0],
                      [0, 0, 0, 0.3, 0, 0],
                      [0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 0.3]])*160
    
    R_att = np.diag(np.ones(4))*40
    
    
    Q_t = np.array([[1e-08, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1e-08, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1e-08, 0],
                    [0, 0, 0, 0, 0, 0.5]])*60
    
    R_t = np.diag(np.ones(3))*5




A_att = np.array([[0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0]])

B_att = np.array([[0, 0, 0, 0],
              [0, 1/I_xx, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1/I_yy, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1/I_zz]])


R_inv_att = np.linalg.inv(R_att)

P_att = solve_lqr(A_att, B_att, Q_att, R_att)

K_att = -np.dot(R_inv_att, np.dot(B_att.T, P_att)) 




A_t = np.array([[0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0]])

B_t = np.array([[0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 1]])/M



R_inv_t = np.linalg.inv(R_t)

P_t = solve_lqr(A_t, B_t, Q_t, R_t)

K_t = -np.dot(R_inv_t, np.dot(B_t.T, P_t)) 


time_int_step = 0.01
T = 1

env = quad(time_int_step, max_timesteps, training=True, euler=0, direct_control=0, T=T, clipped = clipped)
env_plot = plotter(env, True, False)
int_state = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
# state, action = env.reset(int_state)
effort_array = []
for j in range(n_episodes):
    state, action = env.reset()
    effort = 0
    euler_t_ant = env.ang
    for i in range(max_timesteps):
        state_t = np.array([0, env.state[1], 0, env.state[3], 0, env.state[5]]) 
        F = np.dot(K_t, state_t)
 
        phi = env.ang[0]
        theta = env.ang[1]
        psi = env.ang[2]
        theta_t = np.arctan2(F[0], (F[2]+G))
        # theta_t = np.clip(theta_t, -np.pi/2, np.pi/2)
        phi_t = np.arctan2(-F[1]*np.cos(theta_t), (F[2]+G))
        # phi_t = np.clip(phi_t, -np.pi/2, np.pi/2)
        euler_t = np.array([phi_t, theta_t, 0])
        # U_1 = M*(F[2]+G)/(np.cos(env.ang[1])*np.cos(env.ang[0]))
        U_1 = M*(F[2]+G)/(np.cos(theta_t)*np.cos(phi_t))

        
        euler =   env.ang - euler_t 
        
        deuler_t = (euler_t - euler_t_ant)/time_int_step
        
        euler_t_ant = euler_t
        
        d_euler = env.ang_vel
        
        state_att = np.array([euler[0], d_euler[0], euler[1], d_euler[1], euler[2], d_euler[2]])
        action = np.dot(K_att, state_att)
        action[0] = U_1
        # print(action)
        # action = np.clip(action, [M*G/2, -1/3, -1/3, -1/3], [1.5*M*G, 1/3, 1/3, 1/3])
        # print(action)
        # print(env.ang, euler_t, )
        # print(F)
        # print(action)
        env_plot.add(np.zeros(13))
        effort += np.sum(np.abs(env.step_effort))
        _, _, done = env.step(action)
        
        memory_step = np.concatenate((env.state[1:6:2], env.ang, env.ang_vel, env.step_effort))
        memory_array[j, i, :] = memory_step

    env_plot.plot()
    
clipped_str = '' if clipped else '_not_clipped'
np.save('./classical_controller_results/lqr_log'+clipped_str, memory_array)