import numpy as np

class Memory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.index = 0
        self.actions = np.empty([batch_size, 3])
        self.states = np.empty([batch_size, 3, 4, 160, 160])
        self.logprobs = np.empty([batch_size, 1])
        self.rewards = np.empty([batch_size])
        self.is_terminals = np.empty([batch_size])
        self.sens = np.empty([batch_size, 75])
        self.last_conv = np.empty([batch_size])
        self.state_value = np.empty([batch_size])
        
    def clear_memory(self):
        self.index = 0
        self.actions = np.empty([self.batch_size, 3])
        self.states = np.empty([self.batch_size, 3, 4, 160, 160])
        self.logprobs = np.empty([self.batch_size, 1])
        self.rewards = np.empty([self.batch_size, 1])
        self.is_terminals = np.empty([self.batch_size, 1])
        self.sens = np.empty([self.batch_size, 75])
        self.last_conv = np.empty([self.batch_size, 1])
        self.state_value = np.empty([self.batch_size, 1])
    
    def append_memory_rt(self, reward, terminal):
        self.rewards[self.index] = reward
        self.terminal[self.index] = terminal
        self.index += 1
        
    def append_memory_as(self, actions, states, logprobs, sens, state_value):
        self.actions[self.index] = actions
        self.states[self.index] = states
        self.logprobs[self.index] = logprobs
        self.sens[self.index] = sens
        self.state_value[self.index] = state_value
        
        
    def close_memory(self):
        self.actions = np.resize(self.actions, [self.index, 3])
        self.states = np.resize(self.states, [self.index, 3, 4, 160, 160])
        self.logprobs = np.resize(self.logprobs, [self.index, 1])
        self.rewards = np.resize(self.rewards, [self.index])
        self.is_terminals = np.resize(self.is_terminals, [self.index])
        self.sens = np.resize(self.sens, [self.index, 75])
        self.last_conv = np.resize(self.last_conv, [self.index])
        self.state_value = np.resize(self.state_value, [self.index])
        