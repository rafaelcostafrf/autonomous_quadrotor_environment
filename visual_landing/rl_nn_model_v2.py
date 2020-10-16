import torch
import torch.nn as nn
import torch.nn.functional as F
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
H0 = 64
H1 = 48
H2 = 48
SENS_SIZE = 75
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class conv_forward(nn.Module):
    def __init__(self, T):
        super(conv_forward, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels = T, out_channels = 64, kernel_size=7, stride=2, padding = 0)       
        self.conv_11 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=7, stride=2, padding = 0) 
        self.conv_2 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding = 0)
        self.conv_3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1, padding = 0)
        self.fc_1 = nn.Linear(256*8**2, H0)
        
    def forward(self, x):
        # print(x.size())
        x = F.max_pool2d(F.relu(self.conv_1(x)), 2)
        # print(x.size())
        x = F.relu(self.conv_11(x))
        x = F.relu(self.conv_2(x))
        # print(x.size())
        x = F.relu(self.conv_3(x))
        # print(x.size())
        x = F.relu(self.fc_1(torch.flatten(x, start_dim=1)))
        # print(x.size())
        return x


class actor_nn(nn.Module):
    def __init__(self):
        super(actor_nn, self).__init__()
        
        self.fc_1 = nn.Linear(H0, H1)
        
        self.fc_sens = nn.Linear(SENS_SIZE, H1)
        
        self.fc_2 = nn.Linear(2*H1, H2)
        self.fc_3 = nn.Linear(H2, 3)
        
    def forward(self, x, sens):
        x = torch.tanh(self.fc_1(x))

        sens_out = torch.tanh(self.fc_sens(sens))

        x = torch.cat((x, sens_out), dim=1)

        x = torch.tanh(self.fc_2(x))
        x = torch.tanh(self.fc_3(x))
        return x
    
class critic_nn(nn.Module):
    def __init__(self):
        super(critic_nn, self).__init__()
        
        self.fc_1 = nn.Linear(H0, H1)
        
        self.fc_sens = nn.Linear(SENS_SIZE, H1)
        
        self.fc_2 = nn.Linear(2*H1, H2)
        self.fc_3 = nn.Linear(H2, 1)
        
    def forward(self, x, sens):
        x = torch.tanh(self.fc_1(x))
        
        sens_out = torch.tanh(self.fc_sens(sens))

        x = torch.cat((x, sens_out), dim=1)

        x = torch.tanh(self.fc_2(x))
        x = self.fc_3(x)
        return x 
    
             
class ActorCritic(nn.Module):
    def __init__(self, T, action_dim, action_std, child = False):
        super(ActorCritic, self).__init__()        
        if child:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv_forward = conv_forward(T).to(self.device)
        self.actor_nn = actor_nn().to(self.device)
        self.critic_nn = critic_nn().to(self.device)
        
        self.action_var = torch.full((action_dim,), action_std*action_std, dtype=torch.float).to(self.device)
        
    def forward(self, image, sens):
        conv_output = self.conv_forward(image)
        actor_output = self.actor_nn(conv_output, sens)
        critic_output = self.critic_nn(conv_output, sens)
        return actor_output, critic_output

    def act(self, state, sens, memory):

        action_mean, _ = self.forward(state, sens)

        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if len(memory.states) == 0:
            memory.states = np.array(state.detach().cpu().numpy())
            memory.actions = np.array(action.detach().cpu().numpy())
            memory.logprobs = np.array(action_logprob.detach().cpu().numpy()) 
            memory.sens = np.array(sens.detach().cpu().numpy())
        else:
            memory.states = np.append(memory.states, state.detach().cpu().numpy(), axis=0)
            memory.actions = np.append(memory.actions, action.detach().cpu().numpy(), axis=0)
            memory.logprobs = np.append(memory.logprobs, action_logprob.detach().cpu().numpy(), axis=0)
            memory.sens = np.append(memory.sens, sens.detach().cpu().numpy(), axis=0)

        action = action.detach().cpu().numpy().flatten()       
        
        return action
    
    def evaluate(self, state, sens, action):   
        # print(state.size())
        action_mean, state_value = self.forward(state, sens)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy