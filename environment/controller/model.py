import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

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

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda')
class ActorCritic(nn.Module):
    def __init__(self, N, state_dim, action_dim, action_std):
        h1=N
        h2=N
        # print("Tamanho da Rede: {:d}".format(h1))
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, action_dim),
                nn.Tanh()
                ).to(device)
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, 1)
                ).to(device)
        self.action_var = torch.full((action_dim,), action_std*action_std, dtype=torch.float).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        value = self.critic(state)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state.detach())
        memory.actions.append(action.detach())
        memory.logprobs.append(action_logprob.detach())  
        memory.values.append(value.detach())
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
class ActorCritic_old(nn.Module):    
    def __init__(self, state_dim, action_dim, action_std):
        self.device = torch.device('cuda')
        h1=64
        h2=64
        super(ActorCritic_old, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, action_dim),
                nn.Tanh()
                ).to(device)
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, 1)
                ).to(device)
        self.std = torch.nn.Parameter(torch.tensor(action_std))
        self.action_var = torch.full((action_dim,), 1, device = self.device, dtype=torch.float)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)

        cov_mat = torch.diag(self.action_var).to(self.device)*self.std*self.std

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state.detach())
        memory.actions.append(action.detach())
        memory.logprobs.append(action_logprob.detach())  

        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)*self.std*self.std
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(state)
        print(state_value.size())
        return action_logprobs, torch.squeeze(state_value), dist_entropy