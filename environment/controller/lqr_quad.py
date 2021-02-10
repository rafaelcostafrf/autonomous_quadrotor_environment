import sys
sys.path.append('/home/rafaelcostaf/mestrado/quadrotor_environment/')

import numpy as np 
from scipy.linalg import solve_continuous_are as solve_lqr
from environment.quadrotor_env import quad, plotter


I_xx = 16.83e-3
I_yy = 16.83e-3
I_zz = 28.34e-3
M = 1.03
G = 9.82

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

Q_att = np.diag(np.ones(6))
R_att = np.diag(np.ones(4))*20
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


Q_t = np.array([[0.0001, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0.0001, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0.0001, 0],
                [0, 0, 0, 0, 0, 1]])

R_t = np.diag(np.ones(3))*0.3
R_inv_t = np.linalg.inv(R_t)

P_t = solve_lqr(A_t, B_t, Q_t, R_t)

K_t = -np.dot(R_inv_t, np.dot(B_t.T, P_t)) 


time_int_step = 0.01
max_timesteps = 1000
T = 5

env = quad(time_int_step, max_timesteps, training=False, euler=0, direct_control=0, T=T)
env_plot = plotter(env, True, False)
int_state = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
state, action = env.reset(int_state)
# state, action = env.reset()

euler_t_ant = np.zeros(3)
for i in range(max_timesteps):
    state_t = env.state[0:6]
    F = np.dot(K_t, state_t)
    U_1 = M*(F[2]+G)
    
    theta_t = np.arctan(F[0]/(F[2]+G))
    phi_t = np.arctan(-F[1]/(F[2]+G)*np.cos(theta_t))
    euler_t = np.array([phi_t, theta_t, 0])
    deuler_t = (euler_t - euler_t_ant)/time_int_step
    euler_t_ant = euler_t
    
    euler = env.ang - euler_t
    d_euler = env.ang_vel - deuler_t
        
    state_att = np.array([euler[0], d_euler[0], euler[1], d_euler[1], euler[2], d_euler[2]])
    action = np.dot(K_att, state_att)
    action[0] = U_1
    
    env_plot.add(np.zeros(13))
    _, _, done = env.step(action)
    if done:
        break
env_plot.plot()
    