import torch
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from visual_landing.memory_leak import debug_gpu
import torchvision.models as tvmodels
import numpy as np
import torchvision.models as models

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
H0 = 1024
H1 = 256
H11 = 64
H2 = 128
SENS_SIZE = 75
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def plot_conv(x, name):
    for channel in x[0]:
        image = ((channel-channel.mean())/channel.std()).detach().cpu().numpy()
    cv.imshow('conv_in_'+str(name), image)
    cv.waitKey(1)
    return

def plot_conv3D(x, name):
    for channel in x[0, 0]:
        image = ((channel-channel.min())/(channel.max()-channel.min()+1e-8)).detach().cpu().numpy()
    cv.imshow('conv_in_'+str(name), image)
    cv.waitKey(1)
    return

class conv_forward(nn.Module):
    def __init__(self, T):
        super(conv_forward, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=(7,7), stride=(1,1), padding=(3, 3))
        self.conv_2 = nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size=(7,7), stride=(1,1), padding=(3, 3))
        self.conv_3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=(5,5), stride=(1,1), padding=(2, 2))
        self.conv_4 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size=(2,2), stride=(1,1), padding=(1, 1))
        self.fc_1 = nn.Linear(128*6**2, H0)
        
    def forward(self, x):

        x = F.elu(self.conv_1(x))
        x = torch.max_pool2d(x, 2)
        # plot_conv(x, 1)

        x = F.elu(self.conv_2(x))
        x = torch.max_pool2d(x, 2)
        # plot_conv(x, 2)
        

        x = F.elu(self.conv_3(x))
        x = torch.max_pool2d(x, 2)
        # plot_conv(x, 3)
        
        # x = torch.tanh(self.conv_4(x))
        # x = torch.max_pool2d(x, 2)
        # plot_conv(x, 4)
        # print(x.size())
        x = F.elu(self.fc_1(torch.flatten(x, start_dim=1)))

        return x

class conv3D_forward(nn.Module):
    def __init__(self, T, child):
        super(conv3D_forward, self).__init__()
        self.mother = not(child)
        self.conv_1 = nn.Conv3d(in_channels = 3, out_channels = 32, kernel_size=(2,8,8), stride=(1,2,2), padding=(0, 3, 3))
        self.conv_2 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size=(2,4,4), stride=(1,2,2), padding=(0, 1, 1))
        self.conv_3 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size=(2,3,3), stride=(1,1,1), padding=(0, 1, 1))
        self.fc_1 = nn.Linear(64*1*10**2, H0)
        
    def forward(self, x):
        x = torch.tanh(self.conv_1(x))
        # x = torch.max_pool3d(x, (1,2,2))
        # if self.mother:
        #     plot_conv3D(x, 1)

        x = torch.tanh(self.conv_2(x))
        x = torch.max_pool3d(x, (1,2,2))
        # if self.mother:
        #     plot_conv3D(x, 2)
        

        x = torch.tanh(self.conv_3(x))
        x = torch.max_pool3d(x, (1,2,2))
        # if self.mother:
        #     plot_conv3D(x, 3)

        # print(x.size())        
        x = self.fc_1(torch.flatten(x, start_dim=1))

        return x


class actor_nn(nn.Module):
    def __init__(self, T, child=False):
        super(actor_nn, self).__init__()  
        self.conv_ac = conv3D_forward(T, child)
        self.fc_1 = nn.Linear(H0+SENS_SIZE, H1)                
        self.fc_2 = nn.Linear(H1, H2)
        self.fc_21 = nn.Linear(H2, H2)
        self.fc_3 = nn.Linear(H2, 3)
        
    def forward(self, image, sens):
        # print('Actor')
        x = self.conv_ac(image)
        x = torch.cat((x, sens), dim=1)
        x = torch.tanh(self.fc_1(x))
        x = torch.tanh(self.fc_2(x))        
        # x = F.elu(self.fc_21(x))        
        x = torch.tanh(self.fc_3(x))
        # print(x)
        return x
    
class critic_nn(nn.Module):
    def __init__(self, T, child=False):
        super(critic_nn, self).__init__() 
        
        self.conv_ct = conv3D_forward(T, child)
        
        self.fc_1 = nn.Linear(H0+SENS_SIZE+3, H1)                
        self.fc_2 = nn.Linear(H1, H2)
        # self.fc_21 = nn.Linear(H2, H2)
        self.fc_3 = nn.Linear(H2, 1)
        
    def forward(self, image, sens, action):
        # print('Critic')
        # print(x)
        x = self.conv_ct(image)
        x = torch.cat((x, sens, action), dim=1)
        x = torch.tanh(self.fc_1(x))
        x = torch.tanh(self.fc_2(x))   
        # x = F.elu(self.fc_21(x))   
        x = self.fc_3(x)
        # print(x)
        return x 
 
    
             
class ActorCritic(nn.Module):
    def __init__(self, T, action_dim, action_std, child = False):
        super(ActorCritic, self).__init__()        
        if child:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # self.device = torch.device("cpu")
       
        self.actor_nn = actor_nn(T, child).to(self.device)
        self.critic_nn = critic_nn(T, child).to(self.device)
        
        self.action_var = torch.full((action_dim,), action_std*action_std, device = self.device, dtype=torch.float)
        self.action_log_std = torch.full((1, action_dim),action_std, device=self.device)
    
    def __get_dist(self, action_mean):
        action_log_std = self.action_log_std.expand_as(action_mean).to(self.device)
        return torch.distributions.Normal(action_mean, action_log_std)
        
     
    def forward(self, image, sens):
        actor_output = self.actor_nn(image, sens)
        return actor_output

    def critic(self, image, sens, action):
        critic_output = self.critic_nn(image, sens, action)        
        return critic_output
        
        
        
    def act(self, state, sens, last_conv, memory):

        action_mean = self.forward(state, sens)

    
        dist = self.__get_dist(action_mean)
        action = dist.sample()
        
        state_value = self.critic(state, sens, action)
        
        
        action_logprob = dist.log_prob(action).sum(-1)
        memory.append_memory_as(action.detach().cpu().numpy(), state.detach().cpu().numpy(), action_logprob.detach().cpu().numpy(), sens.detach().cpu().numpy(), state_value.detach().cpu().numpy())
            
        action = action.detach().cpu().numpy().flatten()       
        
        return action
    
    def evaluate(self, state, sens, action, last_conv):   
        action_mean = self.forward(state, sens)

        dist = self.__get_dist(action_mean)
        
        action_logprobs = dist.log_prob(action).sum(-1)
        dist_entropy = dist.entropy().sum(-1)
        state_value = self.critic(state, sens, action)
        return action_logprobs, torch.squeeze(state_value), dist_entropy