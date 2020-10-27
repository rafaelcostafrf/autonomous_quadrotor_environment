import sys
sys.path.append('/home/rafael/mestrado/quadrotor_environment/')

import torch
import torch.nn as nn
import time
import numpy as np
from visual_landing.rl_nn_model_v2 import ActorCritic

from random import sample

import os

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    PPO deep learning training algorithm. 
"""


## HYPERPARAMETERS - CHANGE IF NECESSARY ##
lr = 0.0005
action_std = 0.1
K_epochs = 80

eps_clip = 0.2
gamma = 0.99
betas = (0.9, 0.999)
DEBUG = 0
BATCH_SIZE = 256


class PPO:
    def __init__(self, action_dim, child, T):
        if child:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.running_mean = 0
        self.running_n = 0
        self.running_std = 0
        self.M2 = 0
        self.loss_memory = []
        self.loss_memory_step = []
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(T, action_dim, action_std, child)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(T, action_dim, action_std, child)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        try:
            self.policy.load_state_dict(torch.load('./PPO_landing.pth', map_location=self.device))
            self.policy_old.load_state_dict(torch.load('./PPO_landing_old.pth', map_location=self.device))
            print('Saved Landing Policy loaded')
        except:
            torch.save(self.policy.state_dict(), './PPO_landing.pth')
            torch.save(self.policy_old.state_dict(), './PPO_landing_old.pth')
            print('New Landing Policy generated')
            pass
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, sens, last_conv, memory):
        state = torch.Tensor(state).to(self.device).detach()
        network_input = state.unsqueeze(0)

        out = self.policy_old.act(network_input, sens, last_conv, memory)
        return out, last_conv
    
    def running_stats(self, new_samples):
        for x in new_samples:
            delta = x-self.running_mean
            self.running_n += 1
            self.running_mean += delta/self.running_n
            self.M2 += delta*(x-self.running_mean)
        self.running_std = torch.sqrt(self.M2/(self.running_n-1))
        # print(self.running_n, self.running_mean, self.running_std)
    
       
        
        
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
        # self.running_stats(rewards)
        # rewards = (rewards - self.running_mean) / (self.running_std + 1e-5)

        # convert list to tensor
        old_states = torch.tensor(memory.states).detach().to(self.device)
        old_actions = torch.tensor(memory.actions).detach().to(self.device)
        old_logprobs = torch.tensor(memory.logprobs).detach().to(self.device)
        old_sens = torch.tensor(memory.sens).detach().to(self.device)
        old_conv = torch.tensor(memory.last_conv).detach().to(self.device)
        with torch.no_grad():
            _, state_values_adv, _ = self.policy_old.evaluate(old_states, old_sens, old_actions, old_conv)
        advantages = rewards - state_values_adv.detach()
        # advantages = (advantages - advantages.mean())/advantages.std()
        # memory.clear_memory()
    
        
        # print('Out 2 Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
               # Optimize policy for K epochs:
        for step in range(self.K_epochs): 
            self.optimizer.zero_grad()
            rand_batch = torch.randperm(len(old_states))[:BATCH_SIZE]
            old_states_sp = old_states[rand_batch]
            old_actions_sp = old_actions[rand_batch]
            old_logprobs_sp = old_logprobs[rand_batch]
            old_sens_sp = old_sens[rand_batch]
            old_conv_sp = old_conv[rand_batch]
            rewards_sp = rewards[rand_batch]
            advantages_sp = advantages[rand_batch]
            
            print('\rTraining Progress: {:.2%}'.format(step/self.K_epochs), end='          ')
            # print('----------')
            # Evaluating old actions and values :
            # print('Before Log Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000))     
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_sp, old_sens_sp, old_actions_sp, old_conv_sp)
            # print(logprobs.size(), state_values.size(), dist_entropy.size())
            # print('Before Ratios Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000))     
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs_sp.detach())
            # print('Before Advantages Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
            # Finding Surrogate Loss:

            surr1 = ratios * advantages_sp
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_sp
            # print('Before Loss Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(rewards_sp, state_values) - 0.01*dist_entropy
            # print('Before Gradient Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
            # take gradient step

            loss.mean().backward()
            self.optimizer.step()            
            self.loss_memory_step.append(loss.detach().cpu().numpy())
            
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
        self.loss_memory.append(np.mean(self.loss_memory_step))
        self.loss_memory_step=[]
        self.policy_old.load_state_dict(self.policy.state_dict())        

        torch.save(self.policy.state_dict(), './PPO_landing.pth')
        torch.save(self.policy_old.state_dict(), './PPO_landing_old.pth')

        print('Policy Saved      ', end='')
