
import sys
sys.path.append('/home/rafaelcostaf/mestrado/quadrotor_environment/')
from environment.quadrotor_env import quad, plotter
import numpy as np
import torch
from environment.controller.dl_auxiliary import dl_in_gen
from environment.controller.model import ActorCritic
import time

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    PPO testing algorithm (no training, only forward passes)
"""

id_list = ['nn_solved_256_200_aa5a72b63e194ad39dc6b82fc50d4389.pth',
           'nn_solved_128_200_6b4cf43edd934cba9e237c5d6d5a5476.pth',
           'nn_solved_64_200_f30ea177ee4a42e6830c68decdae236d.pth',
           'nn_solved_32_200_081497645ecf4f2795f558b1c9c1fbc8.pth',
           'nn_solved_16_200_b0d209d3630c4ddaab187bf776a13faf.pth']

N_list = [256, 128, 64, 32, 16]





time_int_step = 0.01
max_timesteps = 500
T = 5
N = 128
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
env = quad(time_int_step, max_timesteps, training=False, euler=0, direct_control=1, T=T)
state_dim = 75

dl_aux = dl_in_gen(5, 13, 4)

time_array = []
    
for k, (N, uid) in enumerate(zip(N_list, id_list)):
    time_array_ind = []
    
    policy = ActorCritic(N, state_dim, action_dim=4, action_std=0.01, fixed_std = True).to(device)
    policy.load_state_dict(torch.load('/home/rafaelcostaf/mestrado/quadrotor_environment/environment/controller/solved/'+uid,map_location=device))    
    
    # DO ONE RANDOM EPISODE
    
    nome = 'ramp_unitario_XYZ'
    
    episodes = 100

    
    for j in range(episodes):
        i = 0
        t = 0
        dl_aux.reset()
        state, action = env.reset()    
        in_nn = dl_aux.dl_input(state, action)
        done = False    
        while not done and i < max_timesteps:
            t+=time_int_step
            i+= 1
            time_init = time.time()
            action = policy.actor(torch.FloatTensor(in_nn).to(device)).cpu().detach().numpy()
            time_array_ind.append(time.time()-time_init)
            state, _, done = env.step(action)
            in_nn = dl_aux.dl_input(state, np.array([action]))

    print(str(N)+' '+str(np.mean(time_array_ind)))
