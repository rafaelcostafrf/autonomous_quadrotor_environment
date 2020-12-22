import sys
from environment.quadrotor_env import quad, plotter
import numpy as np
import torch
from environment.controller.dl_auxiliary import dl_in_gen
from environment.controller.model import ActorCritic

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    PPO testing algorithm (no training, only forward passes)
"""

time_int_step = 0.01
max_timesteps = 1000
T = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = quad(time_int_step, max_timesteps, euler=0, direct_control=1, T=T)
state_dim = 75
policy = ActorCritic(state_dim, action_dim=4, action_std=0).to(device)
dl_aux = dl_in_gen(5, 13, 4)
env_plot = plotter(env, velocity_plot=True)

#LOAD TRAINED POLICY
try:
    policy.load_state_dict(torch.load('environment/controller/PPO_continuous_solved_drone.pth',map_location=device))
    print('Saved policy loaded')
except:
    print('Could not load policy')
    sys.exit(1)

T_ERROR = np.array([1])
ERROR = np.array([[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]])
INITIAL_STATE = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

# DO ONE RANDOM EPISODE
state, action = env.reset(INITIAL_STATE)
env_plot.add()
in_nn = dl_aux.dl_input(state, action)
done = False
i_alvo=0
t=0
while not done:
    t+=time_int_step
    if i_alvo+1 <= len(T_ERROR):
        if t > T_ERROR[i_alvo]:
            env.previous_state = ERROR[i_alvo]
            i_alvo += 1
    action = policy.actor(torch.FloatTensor(in_nn).to(device)).cpu().detach().numpy()
    state, _, done = env.step(action)
    in_nn = dl_aux.dl_input(state, np.array([action]))
    env_plot.add()
env_plot.plot()
