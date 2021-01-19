import numpy as np

class Memory:
    def __init__(self, batch_size, image_size, image_time, image_channels):
        self.image_prop = [image_size, image_time, image_channels]
        batch_size = batch_size + 512
        self.batch_size = batch_size
        self.index = 0
        self.actions = np.zeros([batch_size, 3])
        self.states = np.zeros([batch_size, self.image_prop[2], self.image_prop[1], self.image_prop[0][0], self.image_prop[0][0]])
        self.logprobs = np.zeros([batch_size, 1])
        self.rewards = np.zeros([batch_size])
        self.is_terminals = np.zeros([batch_size])
        self.sens = np.zeros([batch_size, 75])
        self.last_conv = np.zeros([batch_size])
        self.state_value = np.zeros([batch_size])
        
    def clear_memory(self):
        self.index = 0
        self.actions = np.zeros([self.batch_size, 3])
        self.states = np.zeros([self.batch_size, self.image_prop[2], self.image_prop[1], self.image_prop[0][0], self.image_prop[0][0]])
        self.logprobs = np.zeros([self.batch_size, 1])
        self.rewards = np.zeros([self.batch_size, 1])
        self.is_terminals = np.zeros([self.batch_size, 1])
        self.sens = np.zeros([self.batch_size, 75])
        self.last_conv = np.zeros([self.batch_size, 1])
        self.state_value = np.zeros([self.batch_size, 1])
    
    def append_memory_rt(self, reward, terminal):
        self.rewards[self.index] = reward
        self.is_terminals[self.index] = terminal
        self.index += 1
        
    def append_memory_as(self, actions, states, logprobs, sens, state_value):
        self.actions[self.index] = actions
        self.states[self.index] = states
        self.logprobs[self.index] = logprobs
        self.sens[self.index] = sens
        self.state_value[self.index] = state_value
        # print(self.actions[self.index-2:self.index+2], self.rewards[self.index-2:self.index+2], self.is_terminals[self.index-2:self.index+2])
        
    def close_memory(self):
        self.actions = np.resize(self.actions, [self.index, 3])
        self.states = np.resize(self.states, [self.index, self.image_prop[2], self.image_prop[1], self.image_prop[0][0], self.image_prop[0][0]])
        self.logprobs = np.resize(self.logprobs, [self.index, 1])
        self.rewards = np.resize(self.rewards, [self.index])
        self.is_terminals = np.resize(self.is_terminals, [self.index])
        self.sens = np.resize(self.sens, [self.index, 75])
        self.last_conv = np.resize(self.last_conv, [self.index])
        self.state_value = np.resize(self.state_value, [self.index])
        