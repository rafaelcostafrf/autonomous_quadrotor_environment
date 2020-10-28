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
        
    def print_memory(self):
        print(np.shape(self.actions), np.shape(self.states), np.shape(self.rewards))
        try:
            print(self.actions[-1], self.states[-1], self.rewards[-1])
        except:
            return

        
