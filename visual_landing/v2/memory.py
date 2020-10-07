import numpy as np

class Memory:
    def __init__(self):
        self.actions = np.array([])
        self.states = np.array([])
        self.logprobs = np.array([])
        self.rewards = np.array([])
        self.is_terminals = np.array([])
    
    def clear_memory(self):
        self.actions = np.array([])
        self.states = np.array([])
        self.logprobs = np.array([])
        self.rewards = np.array([])
        self.is_terminals = np.array([])

        
