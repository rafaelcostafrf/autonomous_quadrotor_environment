import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from visual_landing.memory_leak import debug_gpu
import numpy as np
"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    PPO neural network model
    hidden layers has 64 neurons
"""

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, child = False):
        if child:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h1=64
        h2=64
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.conv1 =  nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4, padding = 2),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(32, 32, kernel_size=4, stride=2, padding = 1),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
                )
        
        self.conv2 =  nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4, padding = 2),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(32, 32, kernel_size=4, stride=2, padding = 1),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
                )
        
        self.actor = nn.Sequential(
                self.conv1,
                nn.Linear(32*5**2, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, 3),
                nn.Tanh(),
            ) 
        # critic
        self.critic = nn.Sequential(
                self.conv2,
                nn.Linear(32*5**2, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std, dtype=torch.float).to(self.device)
        
    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):

        action_mean = self.actor(state)
        
        cov_mat = torch.diag(self.action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states = np.append(memory.states, state.detach().cpu().numpy()[0])
        memory.actions = np.append(memory.actions, action.detach().cpu().numpy()[0])
        memory.logprobs = np.append(memory.logprobs, action_logprob.detach().cpu().numpy()[0])
        
        return action.detach()
    
    def evaluate(self, state, action):   

        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy