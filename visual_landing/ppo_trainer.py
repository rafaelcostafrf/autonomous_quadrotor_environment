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
lr_ac = 0.0001
lr_ct = 0.0001

action_std = 0.15
K_epochs = 8

eps_clip = 0.1
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
            # self.device = torch.device("cpu")
        
        self.running_mean = 0
        self.running_n = 0
        self.running_std = 0
        self.M2 = 0
        self.critic_loss_memory = []
        self.actor_loss_memory = []
        self.critic_epoch_loss = []
        self.actor_epoch_loss = []
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(T, action_dim, action_std, child)
        self.optimizer_ac = torch.optim.Adam(self.policy.actor_nn.parameters(), lr=lr_ac, betas=betas)
        self.optimizer_ct = torch.optim.Adam(self.policy.critic_nn.parameters(), lr=lr_ct, betas=betas)
        
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
    
    def optimizer_step(self, old_states, old_actions, old_logprobs, old_sens, old_conv, rewards, advantages, rand_batch):
        self.optimizer_ct.zero_grad()
        self.optimizer_ac.zero_grad()
        
        old_states_sp = old_states[rand_batch].detach()
        old_actions_sp = old_actions[rand_batch].detach()
        old_logprobs_sp = old_logprobs[rand_batch].detach()
        old_sens_sp = old_sens[rand_batch].detach()
        old_conv_sp = old_conv[rand_batch].detach()
        rewards_sp = rewards[rand_batch].detach()
        advantages_sp = torch.unsqueeze(advantages[rand_batch].detach(), 1)
        
        # print(old_states_sp.size(), old_actions_sp.size(), old_logprobs_sp.size(), old_sens_sp.size(), old_conv_sp.size(), rewards_sp.size(), advantages_sp.size())
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_sp, old_sens_sp, old_actions_sp, old_conv_sp)


        ratios = (logprobs-old_logprobs_sp.flatten()).exp()
        surr1 = ratios * advantages_sp.flatten()
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_sp.flatten()
        actor_loss = torch.min(surr1, surr2)
        entropy_loss = dist_entropy 
        # print(state_values, rewards_sp)
        critic_loss = 0.5*self.MseLoss(state_values, rewards_sp)
        # print(actor_loss, entropy_loss)
        loss_ac = -actor_loss - 0.01*entropy_loss
        
        loss_ac.mean().backward()
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer_ac.step()
        critic_loss.mean().backward()
        self.optimizer_ct.step()

        self.critic_epoch_loss.append(critic_loss.mean().detach().cpu().numpy())
        self.actor_epoch_loss.append(actor_loss.mean().detach().cpu().numpy())
    
    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0
        lmbda = 0.99
        gmma = 0.99
        for i in reversed(range(len(rewards))):
            if i == len(rewards):
                delta = rewards[i] - values[i]
            else:
                delta = rewards[i] + gmma * values[i + 1] * masks[i] - values[i]
            gae = delta + gmma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])
    
        adv = torch.tensor(returns).detach().to(self.device) - values[:-1].clone().detach()
        returns = torch.tensor(returns).to(self.device, dtype=torch.float)
        return returns, (adv - adv.mean()) / (adv.std() + 1e-10)
    
    def update(self, memory):
        # print('Out Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
        # Monte Carlo estimate of rewards:
        # rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)

        # # Normalizing the rewards:
        # rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        # self.running_stats(rewards)
        # rewards = (rewards - self.running_mean) / (self.running_std + 1e-5)


        # convert list to tensor
        old_states = torch.tensor(memory.states).detach().to(self.device, dtype=torch.float)
        old_actions = torch.tensor(memory.actions).detach().to(self.device, dtype=torch.float)
        old_logprobs = torch.tensor(memory.logprobs).detach().to(self.device, dtype=torch.float)
        old_sens = torch.tensor(memory.sens).detach().to(self.device, dtype=torch.float)
        old_conv = torch.tensor(memory.last_conv).detach().to(self.device, dtype=torch.float)
        state_values = torch.tensor(memory.state_value).detach().to(self.device, dtype=torch.float)
        state_values_ex = torch.cat((state_values, torch.tensor([0]).to(self.device)))
        rewards, advantages = self.get_advantages(state_values_ex, np.logical_not(memory.is_terminals), torch.tensor(memory.rewards).to(self.device))
        # advantages = rewards - state_values.detach()

        # print(advantages[-20:])
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5).detach()

        # memory.clear_memory()
        # print(old_states.size(), old_actions.size(), state_values.size(), old_logprobs.size())
        
        # print('Out 2 Step Training Memory: {:.2f}Mb'.format(process.memory_info().rss/1000000)) 
               # Optimize policy for K epochs:
        
        for step_j in range(self.K_epochs):
            rand_total = torch.randperm(old_states.size()[0], device=self.device)
            total_range = int(old_states.size()[0]/BATCH_SIZE)
            for step_i in range(total_range): 
                rand_batch = rand_total[step_i*BATCH_SIZE:(step_i+1)*BATCH_SIZE]
                self.optimizer_step(old_states, old_actions, old_logprobs, old_sens, old_conv, rewards, advantages, rand_batch)
            rest_range = old_states.size()[0]%BATCH_SIZE
            if rest_range != 0:                  
                rand_batch = rand_total[-rest_range:]
                self.optimizer_step(old_states, old_actions, old_logprobs, old_sens, old_conv, rewards, advantages, rand_batch)
            
        self.critic_loss_memory.append(np.mean(self.critic_epoch_loss))
        self.actor_loss_memory.append(np.mean(self.actor_epoch_loss))
        self.critic_epoch_loss = []
        self.actor_epoch_loss = []
        self.policy_old.load_state_dict(self.policy.state_dict())  
        torch.save(self.policy.state_dict(), './PPO_landing.pth')
        torch.save(self.policy_old.state_dict(), './PPO_landing_old.pth')
