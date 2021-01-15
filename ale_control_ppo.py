import sys
from environment.quadrotor_env import quad, plotter
import numpy as np
import torch
from environment.controller.dl_auxiliary import dl_in_gen
from environment.controller.model import ActorCritic
import time
import matplotlib
from matplotlib import pyplot as plt

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    PPO testing algorithm (no training, only forward passes)
"""


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "lualatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

time_int_step = 0.01
max_timesteps = 500
end_time = max_timesteps*time_int_step
time_array = np.arange(0, end_time, time_int_step) 
T = 5

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
env = quad(time_int_step, max_timesteps, euler=0, direct_control=1, T=T)
state_dim = 75
policy = ActorCritic(state_dim, action_dim=4, action_std=0).to(device)
dl_aux = dl_in_gen(5, 13, 4)


#LOAD TRAINED POLICY
try:
    policy.load_state_dict(torch.load('environment/controller/PPO_continuous_drone_velocity_seed_128.pth',map_location=device))
    print('Saved policy loaded')
except:
    print('Could not load policy')
    sys.exit(1)

episodes = 200
vel_array = np.zeros([episodes, max_timesteps, 3])
err_perm = np.zeros([episodes, 3])
err_tot = np.zeros([episodes, max_timesteps])
act_tot = np.zeros([episodes, max_timesteps, 4])
act_i = np.zeros([episodes, max_timesteps]) 
# DO ONE RANDOM EPISODE

for i in range(episodes):
    state, action = env.reset()
    in_nn = dl_aux.dl_input(state, action)
    done = False
    j=0
    while j < max_timesteps:     
        action = policy.actor(torch.FloatTensor(in_nn).to(device)).cpu().detach().numpy()
        state, _, done = env.step(action)
        in_nn = dl_aux.dl_input(state, np.array([action]))
        vel_array[i, j, :] = state[0, 1:6:2]
        err_tot[i, j] = np.linalg.norm(state[0, 1:6:2])
        act_tot[i, j, :] = action
        act_i[i, j] = np.linalg.norm(action)
        j+= 1
    err_perm[i, :] = state[0, 1:6:2]

err_mean = np.linalg.norm(err_perm, axis=1)    
cond = np.abs(err_mean)<0.1
P_s = np.sum(cond)/episodes
E_e = np.mean(err_mean[cond])    
err_tot_mean = np.sum(np.linalg.norm(vel_array, axis=2), axis=1)/max_timesteps
err_tot = np.delete(np.linalg.norm(vel_array, axis=2), np.logical_not(cond), axis=0)
err_tot_plot = np.mean(err_tot, axis=0)
err_tot_plot_std = np.std(err_tot, axis=0)
E_v = np.mean(err_tot_mean[cond])    
F_c = np.mean(np.mean(act_i, axis=1)[cond]) 

print('P_s= {:.2%} \t E_e= {:.3e} \t E_v= {:.3e} \t F_c= {:.3e}'.format(P_s, E_e, E_v, F_c) )   

fig, ax = plt.subplots(1)
ax.plot(time_array, err_tot_plot)
ax.fill_between(time_array, err_tot_plot - err_tot_plot_std, err_tot_plot + err_tot_plot_std, alpha=0.5)
ax.set_xlabel('tempo (s)')
ax.set_ylabel('velocidade (m/s)')
ax.grid(True)
plt.savefig('resposta_ale.pgf')
# plt.show()