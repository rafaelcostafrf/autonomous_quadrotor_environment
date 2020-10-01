
from visual_landing.quad_worker import quad_worker
from visual_landing.ppo_worker import ppo_worker
N_WORKERS = 10
EPISODES_UNTIL_TRAIN = 50
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class work_flow():
    
    def __init__(self, render, cv_cam):
        self.MEMORY = Memory()
        self.render = render
        self.cv_cam = cv_cam
        self.reset_workers()
        self.render.taskMgr.add(self.episode_done_check, 'done_check')
        self.done_episodes = 0
        
    def reset_workers(self):
        self.workers = []
        for i in range(N_WORKERS):
            self.workers.append(quad_worker(self.render))
            self.render.taskMgr.setupTaskChain(str(i), numThreads = 1, tickClock = None,
                                   threadPriority = None, frameBudget = None,
                                   frameSync = None, timeslicePriority = None)
            self.render.taskMgr.add(self.workers[i].step, str(i), taskChain = str(i))
  
        self.ppo_worker = ppo_worker(self.render, self.workers, self.cv_cam)
    
        self.render.taskMgr.add(self.ppo_worker.wait_until_ready, 'ppo_wait')
    
    def episode_done_check(self, task):
        done = True
        for worker in self.workers:
            if not worker.visual_done:
               done = False 
        
        if done == True:
            self.done_episodes += N_WORKERS
            for worker in self.workers:
                self.MEMORY.actions.extend(worker.memory.actions)
                self.MEMORY.states.extend(worker.memory.states)
                self.MEMORY.logprobs.extend(worker.memory.logprobs)
                self.MEMORY.rewards.extend(worker.memory.rewards)
                self.MEMORY.is_terminals.extend(worker.memory.is_terminals)
            if self.done_episodes >= EPISODES_UNTIL_TRAIN:
                self.ppo_worker.ldg_policy.update(self.MEMORY)
                self.MEMORY.clear_memory()
            self.reset_workers()
        return task.cont
        