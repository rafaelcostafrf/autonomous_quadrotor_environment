import sys
sys.path.append('/home/rafaelcostaf/mestrado/quadrotor_environment/')
from environment.quadrotor_env import quad, plotter
import numpy as np
import torch
from environment.controller.dl_auxiliary import dl_in_gen
from environment.controller.model import ActorCritic
import time
import glob

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    PPO testing algorithm (no training, only forward passes)
"""
file_name = 'nn_old_solved_128_32000_e8f2b04d78f34fc794e686dcb00944d2.pth'
file = '/home/rafaelcostaf/mestrado/quadrotor_environment/environment/controller/solved/'+file_name

N = 128
time_int_step = 0.01
max_timesteps = 1500
T = 5
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
env = quad(time_int_step, max_timesteps, training=False, euler=0, direct_control=1, T=T)
state_dim = 75
policy = ActorCritic(N, state_dim, action_dim=4, action_std=0.01, fixed_std = True).to(device)
dl_aux = dl_in_gen(5, 13, 4)
env_plot = plotter(env, velocity_plot=True)
#LOAD TRAINED POLICY
policy.load_state_dict(torch.load(file, map_location=device))
    
n_episodes = 500
memory_array = np.zeros([n_episodes, max_timesteps, 13])

for j in range(n_episodes):

    # DO ONE RANDOM EPISODE
    state, action = env.reset()
    dl_aux.reset()
    env_plot.add(np.zeros(13))
    in_nn = dl_aux.dl_input(state, action)
    done = False
    i = 0
    in_nn = dl_aux.dl_input(state, action)
    while i < max_timesteps:
        action = policy.actor(torch.FloatTensor(in_nn).to(device)).cpu().detach().numpy()
        state, _, done = env.step(action)
        in_nn = dl_aux.dl_input(state, np.array([action]))
        env_plot.add(np.zeros(13))

        memory_step = np.concatenate((env.state[1:6:2], env.ang, env.ang_vel, env.step_effort))
        memory_array[j, i, :] = memory_step
        i+= 1

    env_plot.plot()

np.save('/home/rafaelcostaf/mestrado/quadrotor_environment/environment/controller/classical_controller_results/rl_log', memory_array)
