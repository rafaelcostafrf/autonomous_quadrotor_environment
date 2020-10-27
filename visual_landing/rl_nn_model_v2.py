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
H0 = 512
H1 = 512
H11 = 512
H2 = 512
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
        image = ((channel-channel.mean())/channel.std()).detach().cpu().numpy()
    cv.imshow('conv_in_'+str(name), image)
    cv.waitKey(1)
    return

class conv_forward(nn.Module):
    def __init__(self, T):
        super(conv_forward, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=7)       
        self.conv_2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size=5)
        self.conv_3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size=5)
        # self.conv_4 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size=3)
        self.fc_1 = nn.Linear(32*24**2, H0)
        
    def forward(self, x):

        x = torch.tanh(self.conv_1(x))
        x = torch.max_pool2d(x, 2)
        plot_conv(x, 1)

        x = torch.tanh(self.conv_2(x))
        x = torch.max_pool2d(x, 2)
        plot_conv(x, 2)
        

        x = torch.tanh(self.conv_3(x))
        x = torch.max_pool2d(x, 2)
        plot_conv(x, 3)
        
        # x = torch.tanh(self.conv_4(x))
        # x = torch.max_pool2d(x, 2)
        # plot_conv(x, 4)
        # print(x.size())
        x = torch.tanh(self.fc_1(torch.flatten(x, start_dim=1)))

        return x

class conv3D_forward(nn.Module):
    def __init__(self, T):
        super(conv3D_forward, self).__init__()
        
        self.conv_1 = nn.Conv3d(in_channels = 1, out_channels = 32, kernel_size=(2,8,8), stride=(1,2,2))
        self.conv_2 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size=(2,4,4), stride=(1,2,2))
        self.conv_3 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size=(2,3,3), stride=(1,1,1))
        # self.conv_4 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size=3)
        self.fc_1 = nn.Linear(64*1*17**2, H0)
        
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = F.relu(self.conv_1(x))
        # x = torch.max_pool3d(x, (1,2,2))
        # plot_conv3D(x, 1)

        x = F.relu(self.conv_2(x))
        # x = torch.max_pool3d(x, (1,2,2))
        # plot_conv(x, 2)
        

        x = F.relu(self.conv_3(x))
        # x = torch.max_pool3d(x, (1,2,2))
        # plot_conv3D(x, 3)
        
        # x = torch.tanh(self.conv_4(x))
        # x = torch.max_pool2d(x, 2)
        # plot_conv(x, 4)
        # print(x.size())
        x = torch.tanh(self.fc_1(torch.flatten(x, start_dim=1)))

        return x


class actor_nn(nn.Module):
    def __init__(self):
        super(actor_nn, self).__init__()
        
        # self.fc_1 = nn.Linear(H0, H1)
        
        self.fc_sens = nn.Linear(SENS_SIZE, H11)
        
        self.fc_2 = nn.Linear(H1+H11, H2)
        # self.fc_3 = nn.Linear(H2, H2)
        self.fc_4 = nn.Linear(H2, 3)
        
    def forward(self, x, sens):

        # x = torch.tanh(self.fc_1(x))
        
        sens_out = torch.tanh(self.fc_sens(sens))

        x = torch.cat((x, sens_out), dim=1)

        x = torch.tanh(self.fc_2(x))
        # x = torch.tanh(self.fc_3(x))
        x = torch.tanh(self.fc_4(x))
        return x
    
class critic_nn(nn.Module):
    def __init__(self):
        super(critic_nn, self).__init__()
        
        # self.fc_1 = nn.Linear(H0, H1)
        
        self.fc_sens = nn.Linear(SENS_SIZE+3, H11)
        
        self.fc_2 = nn.Linear(H1+H11, H2)
        # self.fc_3 = nn.Linear(H2, H2)
        self.fc_4 = nn.Linear(H2, 1)
        
    def forward(self, x, sens, action):

        # x = torch.tanh(self.fc_1(x))

        sens = torch.cat((sens, action), dim=1)
        sens_out = torch.tanh(self.fc_sens(sens))

        x = torch.cat((x, sens_out), dim=1)

        x = torch.tanh(self.fc_2(x))
        # x = torch.tanh(self.fc_3(x))
        x = self.fc_4(x)
        return x 
    
             
class ActorCritic(nn.Module):
    def __init__(self, T, action_dim, action_std, child = False):
        super(ActorCritic, self).__init__()        
        if child:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.resnet18 = models.resnet18().to(self.device)
        self.conv = conv3D_forward(T).to(self.device)
        self.actor_nn = actor_nn().to(self.device)
        self.critic_nn = critic_nn().to(self.device)
        
        self.action_var = torch.full((action_dim,), action_std*action_std, dtype=torch.float).to(self.device)
        
    def forward(self, image, sens, action):

        # conv_output = self.conv(image)
        # print(image[:, 0].size())
        conv_output = self.conv(image)
        # x_1 = self.conv(torch.unsqueeze(image[:, 1], 1)).detach()
        # x_2 = self.conv(torch.unsqueeze(image[:, 2], 1)).detach()
            
        # conv_output = torch.cat([x_0, x_1, x_2], dim=1)

        actor_output = self.actor_nn(conv_output, sens)
        critic_output = self.critic_nn(conv_output, sens, action)
        
        return actor_output, critic_output

    def act(self, state, sens, last_conv, memory):

        action_mean, _ = self.forward(state, sens, torch.zeros([1, 3]).to(self.device))

        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if len(memory.states) == 0:
            memory.states = np.array(state.detach().cpu().numpy())
            memory.actions = np.array(action.detach().cpu().numpy())
            memory.logprobs = np.array(action_logprob.detach().cpu().numpy()) 
            memory.sens = np.array(sens.detach().cpu().numpy())
            memory.last_conv = np.array(last_conv.detach().cpu().numpy())
        else:
            memory.states = np.append(memory.states, state.detach().cpu().numpy(), axis=0)
            memory.actions = np.append(memory.actions, action.detach().cpu().numpy(), axis=0)
            memory.logprobs = np.append(memory.logprobs, action_logprob.detach().cpu().numpy(), axis=0)
            memory.sens = np.append(memory.sens, sens.detach().cpu().numpy(), axis=0)
            memory.last_conv = np.append(memory.last_conv, last_conv.detach().cpu().numpy(), axis=0)

        action = action.detach().cpu().numpy().flatten()       
        
        return action
    
    def evaluate(self, state, sens, action, last_conv):   
        # print(state.size())
        action_mean, state_value = self.forward(state, sens, action)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy