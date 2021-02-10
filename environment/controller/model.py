import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

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

device = torch.device("cpu")

class ActorCritic(nn.Module):
    def __init__(self, N, state_dim, action_dim, action_std, fixed_std=False):
        h1=N
        h2=N

        super(ActorCritic, self).__init__()

        self.actor =  nn.Sequential(
                nn.Linear(state_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, action_dim),
                nn.Tanh()
                ).to(device)

        self.critic = nn.Sequential(
                nn.Linear(state_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, 1)
                ).to(device)
        self.action_var = torch.full((action_dim,), 1, device = device, dtype=torch.double)
        if fixed_std:
            self.std = torch.nn.Parameter(torch.tensor(action_std), requires_grad = False)
        else:
            self.std = torch.nn.Parameter(torch.tensor(action_std), requires_grad = True)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):

        action_mean = self.actor(state)

        value = self.critic(state)

        cov_mat = torch.diag(self.action_var).to(device)*self.std*self.std



        dist = Normal(action_mean, torch.ones(4)*self.std)

        action = dist.sample()

        action_logprob = dist.log_prob(action)
        memory.states.append(state.detach())
        memory.actions.append(action.detach())
        memory.logprobs.append(action_logprob.detach())  
        memory.values.append(value.detach())

        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)*self.std*self.std

        dist = Normal(action_mean, torch.ones(4)*self.std)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
class ActorCritic_old(nn.Module):
    def __init__(self, N, state_dim, action_dim, action_std):
        h1=N
        h2=N

        super(ActorCritic_old, self).__init__()

        self.actor =  nn.Sequential(
                nn.Linear(state_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, action_dim),
                nn.Tanh()
                ).to(device)

        self.critic = nn.Sequential(
                nn.Linear(state_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, 1)
                ).to(device)
        self.action_var = torch.full((action_dim,), 1, device = device, dtype=torch.double)

        self.std = action_std

        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):

        action_mean = self.actor(state)

        value = self.critic(state)

        cov_mat = torch.diag(self.action_var).to(device)*self.std*self.std



        dist = Normal(action_mean, torch.ones(4)*self.std)

        action = dist.sample()

        action_logprob = dist.log_prob(action)
        memory.states.append(state.detach())
        memory.actions.append(action.detach())
        memory.logprobs.append(action_logprob.detach())  
        memory.values.append(value.detach())

        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)*self.std*self.std

        dist = Normal(action_mean, torch.ones(4)*self.std)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy