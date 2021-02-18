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
N_ARRAY = [16, 32, 64, 128, 256]

for N in N_ARRAY:
    time_int_step = 0.01
    max_timesteps = 500
    T = 5
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    env = quad(time_int_step, max_timesteps, training=False, euler=0, direct_control=1, T=T)
    state_dim = 75
    policy = ActorCritic(N, state_dim, action_dim=4, action_std=0.01, fixed_std = True).to(device)
    dl_aux = dl_in_gen(5, 13, 4)
    env_plot = plotter(env, velocity_plot=True)

    for file in glob.glob('/home/rafaelcostaf/mestrado/quadrotor_environment/environment/controller/solved/nn_old_solved_'+str(N)+'*.pth'):
        #LOAD TRAINED POLICY
        policy.load_state_dict(torch.load(file,map_location=device))
        
        T_ERROR = np.array([0, 1])
        ERROR = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, -5, 0, -5, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        INT_STATE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        RAMP = np.linspace(ERROR[0], ERROR[1], max_timesteps)
        
        # DO ONE RANDOM EPISODE
        state, action = env.reset(INT_STATE)
        dl_aux.reset()
        env_plot.add(INT_STATE)
        in_nn = dl_aux.dl_input(state, action)
        done = False
        target_done = False
        i_alvo=0
        t=0
        i = 0
        nome = str(N)+"  "+file[-9:-4]
        
        degrau = False
        ramp = True
        random = False
        
        
        if random:
            state, action = env.reset()
            env_plot.add(INT_STATE)
            in_nn = dl_aux.dl_input(state, action)
            while not done and i < max_timesteps:
                t+=time_int_step
                i+= 1
                action = policy.actor(torch.FloatTensor(in_nn).to(device)).cpu().detach().numpy()
                state, _, done = env.step(action)
                in_nn = dl_aux.dl_input(state, np.array([action]))
                env_plot.add(-ERROR[i_alvo])        
                if not target_done:
                    if t >= T_ERROR[i_alvo]:
                        if i_alvo+1 == len(ERROR):
                            target_done = True
                        else:
                            i_alvo += 1
            print(env.target_state, env.current_state, env.reward)
            env_plot.plot(nome)
            
            
        if degrau:
            while not done and i < max_timesteps:
                t+=time_int_step
                i+= 1
                action = policy.actor(torch.FloatTensor(in_nn).to(device)).cpu().detach().numpy()
                state, _, done = env.step(action)
                in_nn = dl_aux.dl_input(state+ERROR[i_alvo], np.array([action]))
                env_plot.add(-ERROR[i_alvo])        
                if not target_done:
                    if t >= T_ERROR[i_alvo]:
                        if i_alvo+1 == len(ERROR):
                            target_done = True
                        else:
                            i_alvo += 1
            env_plot.plot(nome)
            
        if ramp:
            while not done and i < max_timesteps:
                ERROR = RAMP[i]
                action = policy.actor(torch.FloatTensor(in_nn).to(device)).cpu().detach().numpy()
                state, _, done = env.step(action)
                in_nn = dl_aux.dl_input(state+ERROR, np.array([action]))
                env_plot.add(-ERROR)  
                t+=time_int_step
                i+= 1
            env_plot.plot(nome)
            