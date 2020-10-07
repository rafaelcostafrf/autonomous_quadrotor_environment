import sys
sys.path.append('/home/rafael/mestrado/quadrotor_environment/')

import torch
import torch.nn as nn
import time
import gc

from visual_landing.nn_model_v2 import ActorCritic
from visual_landing.memory_leak import debug_gpu
import os
import psutil
process = psutil.Process(os.getpid())
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
torch.set_num_threads(4)
PROCESS_TIME = time.time()



## HYPERPARAMETERS - CHANGE IF NECESSARY ##
lr = 0.001
max_timesteps = 1000
action_std = 0.2
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



class PPO:
    def __init__(self, state_dim, action_dim, child = False):
        if child:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std, child).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std, child).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        try:
            self.policy.load_state_dict(torch.load('./PPO_landing.pth', map_location=self.device))
            self.policy_old.load_state_dict(torch.load('./PPO_landing_old.pth', map_location=self.device))
            print('Saved Landing Policy loaded')
        except:
            print('New Landing Policy generated')
            pass
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.Tensor(state).to(self.device).detach()
        network_input = state.unsqueeze(0)
        out = self.policy_old.act(network_input, memory)
        return out
    
    def update(self, memory):
        # print('Out Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        

        # convert list to tensor
        old_states = torch.tensor(memory.states).to(self.device).detach()
        old_actions = torch.tensor(memory.actions).to(self.device).detach()
        old_logprobs = torch.tensor(memory.logprobs).to(self.device).detach()
        memory.clear_memory()
        # print('Out 2 Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
               # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # print('----------')
            # Evaluating old actions and values :
            # print('Before Log Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000))     
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # print('Before Ratios Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000))     
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # print('Before Advantages Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # print('Before Loss Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            # print('Before Gradient Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
            # take gradient step
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()            

            
            # del state_values
            # del dist_entropy
            # del logprobs
            # del ratios
            # del advantages
            # del surr1
            # del surr2
            # del loss
            # print('After Gradient Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
            # print('----------')
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())        

        torch.save(self.policy.state_dict(), './PPO_landing.pth')
        torch.save(self.policy_old.state_dict(), './PPO_landing_old.pth')
        print('Policy Saved      ', end='')
