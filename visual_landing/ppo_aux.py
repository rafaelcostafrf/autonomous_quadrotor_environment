import sys
sys.path.append('/home/rafael/mestrado/quadrotor_environment/')
# from datetime import datetime, date

import torch
import torch.nn as nn
# from torch.distributions import MultivariateNormal
import numpy as np
import time

# from environment.quadrotor_env import quad, plotter
# from dl_auxiliary import dl_in_gen
from visual_landing.nn_model import ActorCritic

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    PPO deep learning training algorithm. 
"""
random_seed = 666
seed = '_velocity_seed_'+str(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
PROCESS_TIME = time.time()



## HYPERPARAMETERS - CHANGE IF NECESSARY ##
lr = 0.001
max_timesteps = 1000
action_std = 0.05
update_timestep = 4000
K_epochs = 80
T = 5


## HYPERPAREMETERS - PROBABLY NOT NECESSARY TO CHANGE ##
action_dim = 4

log_interval = 100
max_episodes = 100000
time_int_step = 0.01
solved_reward = 700
eps_clip = 0.2
gamma = 0.99
betas = (0.9, 0.999)
DEBUG = 0


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPO:
    def __init__(self, state_dim, action_dim):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        try:
            self.policy.load_state_dict(torch.load('./PPO_landing.pth',map_location=device))
            self.policy_old.load_state_dict(torch.load('./PPO_landing_old.pth',map_location=device))
            print('Saved Landing Policy loaded')
        except:
            print('New Landing Policy generated')
            pass
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor([state]).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
               
        # Optimize policy for K epochs:
        for epoch in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            print('\rTraining progress: {:.2%}          '.format(epoch/self.K_epochs),end='')
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())



def evaluate(env, agent, plotter, eval_steps=10):
    n_solved = 0
    rewards = 0
    time_steps = 0
    for i in range(eval_steps):
        state, action = env.reset()
        done = False
        while True:
            time_steps += 1
            network_in = aux_dl.dl_input(state, action)    
            action = agent.policy.actor(torch.FloatTensor(network_in).to(device)).cpu().detach().numpy()  
            state, reward, done = env.step(action)
            action = np.array([action])
            rewards += reward
            if i == eval_steps-1:
                plotter.add()
            if done:
                n_solved += env.solved
                break
    time_mean = time_steps/eval_steps
    solved_mean = n_solved/eval_steps        
    reward_mean = rewards/eval_steps
    plotter.plot()
    return reward_mean, time_mean, solved_mean
    


# # creating environment
# env = quad(time_int_step, max_timesteps, euler=0, direct_control=1, T=T)
# state_dim = 15*T
# plot = plotter(env, True, False)

# #creating reward logger
# if random_seed:
#     print("Random Seed: {}".format(random_seed))
#     torch.manual_seed(random_seed)
#     env.seed(random_seed)
#     np.random.seed(random_seed)

# # creating ppo trainer
# memory = Memory()
# ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
# print(lr,betas)

# # logging variables
# t_since_last_plot = 0
# running_reward = 0
# avg_length = 0
# time_step = 0
# solved_avg = 0
# eval_on_mean = True

# # auxiliar deep learning input generator
# aux_dl = dl_in_gen(T, env.state_size, env.action_size)  
# aux_dl.reset()


# # training loop
# for i_episode in range(1, max_episodes+1):
#     # Resets the environment
#     state, action = env.reset()
    
#     #Prints the progress on console
#     print('\rProgress: '+str(int((i_episode-1)%log_interval/log_interval*100))+'% ',end='\r')
    
#     for t in range(max_timesteps):        
        
#         # Converts the quadrotor past states and actions to neural network input
#         network_in = aux_dl.dl_input(state, action)        
        
#         t_since_last_plot += 1
#         time_step +=1
        
#         # Running policy_old:
#         action = ppo.select_action(network_in, memory)
#         state, reward, done = env.step(action)
#         action = np.array([action])
        
#         # Saving reward and is_terminals:
#         memory.rewards.append(reward)
#         memory.is_terminals.append(done)

#         # update if its time
#         if time_step % update_timestep == 0:
#             ppo.update(memory)
#             memory.clear_memory()
#             time_step = 0
#         running_reward += reward
#         if done:  
#             break
#     avg_length += t
    
#     # save every x episodes
#     if i_episode % 100 == 0:
#         torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format('drone'+seed))
#         torch.save(ppo.policy_old.state_dict(), './PPO_continuous_old_{}.pth'.format('drone'+seed))
        
#     # logging
#     if i_episode % log_interval == 0:
#         reward_avg, time_avg, solved_avg = evaluate(env, ppo, plot, 20)
#         avg_length = int(avg_length/log_interval)
#         running_reward = int((running_reward/log_interval))
#         print('\rEpisode {} \t Avg length: {} \t Avg reward: {:.2f} \t Solved: {:.2f}'.format(i_episode, time_avg, reward_avg, solved_avg))
#         running_reward = 0
#         avg_length = 0
        
#         today = date.today()
#         # dd/mm/YY
#         d1 = today.strftime("%d/%m/%Y")
#         now = datetime.now()
#         current_time = now.strftime("%H:%M:%S")
        
#         file_logger = open('eval_reward_log'+seed+'.txt', 'a')
#         file_logger.write(d1+'\t'+current_time+'\t'+str(i_episode)+'\t'+str(time.time()-PROCESS_TIME)+'\t'+str(reward_avg)+'\t'+str(time_avg)+'\n')
#         file_logger.close()
#         # stop training if avg_reward > solved_reward
#         if i_episode==2000:
#             # reward_avg, time_avg, solved_avg = evaluate(env, ppo, plot, 200)
#             # print('\rRe-evaluation \t Avg length: {} \t Avg reward: {:.2f} \t Solved: {:.2f}'.format(time_avg, reward_avg, solved_avg))
#             # if solved_avg > 0.95 and reward_avg>solved_reward:
#                 print("########## Solved! ##########")
#                 torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format('drone'+seed))
#                 torch.save(ppo.policy_old.state_dict(), './PPO_continuous_old_solved_{}.pth'.format('drone'+seed))
#                 break
        