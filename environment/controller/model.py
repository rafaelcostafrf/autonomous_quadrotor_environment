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
    def __init__(self, N, state_dim, action_dim, action_std, fixed_std):
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
        # cov_mat = torch.diag(self.action_var).to(device)*self.fixed_std*self.fixed_std

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
        
        action_var = self.action_var.expand_as(action_mean)*self.std*self.std
        # action_var = self.action_var.expand_as(action_mean)*self.fixed_std*self.fixed_std
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(state)
        # print(self.std)
        return action_logprobs, torch.squeeze(state_value), dist_entropy