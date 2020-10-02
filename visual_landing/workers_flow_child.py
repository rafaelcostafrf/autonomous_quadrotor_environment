import torch
import time
import sys

from visual_landing.quad_worker import quad_worker
from visual_landing.ppo_worker_child import ppo_worker

# LANDING SETUP
from visual_landing.ppo_aux_child import PPO
device = torch.device("cpu")

N_WORKERS = 1
BATCH_SIZE = 270

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions
        del self.states
        del self.logprobs
        del self.rewards
        del self.is_terminals

class work_flow():
    
    def __init__(self, render, cv_cam, child_number):
        
        self.child_number = child_number
        self.MEMORY = Memory()
        self.render = render
        self.cv_cam = cv_cam
        self.ldg_policy = PPO(3, 3)
        
        for i in range(N_WORKERS):
            self.render.taskMgr.setupTaskChain(str(i), numThreads = 1, tickClock = None,
                                   threadPriority = None, frameBudget = None,
                                   frameSync = None, timeslicePriority = None)

        self.render.taskMgr.setupTaskChain('ppo', numThreads = 1, tickClock = None,
                               threadPriority = None, frameBudget = 1,
                               frameSync = None, timeslicePriority = None)

        self.reset_workers()
        self.render.taskMgr.add(self.episode_done_check, 'done_check')
        self.done_episodes = 0
        
        #CAMERA SETUP
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        self.cv_cam = cv_cam
        self.cv_cam.cam.setPos(0, 0, 0)
        self.cv_cam.cam.setHpr(0, 270, 0)
        self.cv_cam.cam.reparentTo(self.render.quad_model)
       
        
    def reset_workers(self):
        while True:
            f = open('./child_data/'+str(self.child_number)+'.txt', 'r')
            if int(f.read()) == 0:
                break
            else:
                time.sleep(3)
            
            
        self.workers = []
        for i in range(N_WORKERS):
            self.workers.append(quad_worker(self.render))
            self.render.taskMgr.add(self.workers[i].step, 'quad_worker'+str(i), taskChain = str(i))  
        try:
            self.ldg_policy.policy.load_state_dict(torch.load('./PPO_landing.pth', map_location=device))
            self.ldg_policy.policy_old.load_state_dict(torch.load('./PPO_landing_old.pth', map_location=device))
        except:
            print("Could Not Load Father Landing Policy")
            sys.exit()
        self.ppo_worker = ppo_worker(self.render, self.workers, self.cv_cam, self.ldg_policy)     
        self.render.taskMgr.add(self.ppo_worker.wait_until_ready, 'ppo_worker'+str(i) , taskChain = 'ppo')
    

    def episode_done_check(self, task):
        done = True
        child_name = './child_data/'+str(self.child_number)

        for worker in self.workers:
            if not worker.visual_done:
               done = False 
               
        if done == True:
            self.done_episodes += N_WORKERS
            batch_size_percentage = len(self.MEMORY.rewards)/BATCH_SIZE
            print('\rEpisode Completion: {:.2%}'.format(batch_size_percentage),end='         ')
            for worker in self.workers:
                self.MEMORY.actions.extend(worker.memory.actions)
                self.MEMORY.states.extend(worker.memory.states)
                self.MEMORY.logprobs.extend(worker.memory.logprobs)
                self.MEMORY.rewards.extend(worker.memory.rewards)
                self.MEMORY.is_terminals.extend(worker.memory.is_terminals)
            
                
            if len(self.MEMORY.rewards)>=BATCH_SIZE:
                child_name = './child_data/'+str(self.child_number)
                
                torch.save(self.MEMORY.actions, child_name+'actions.tch')            
                torch.save(self.MEMORY.states, child_name+'states.tch')  
                torch.save(self.MEMORY.logprobs, child_name+'logprobs.tch')  
                torch.save(self.MEMORY.rewards, child_name+'rewards.tch')  
                torch.save(self.MEMORY.is_terminals, child_name+'is_terminals.tch')  
                self.MEMORY.clear_memory()
                f = open(child_name+'.txt','w')
                f.write(str(1))
                f.close()
            for worker in self.workers:
                worker.memory.clear_memory()    
            self.reset_workers()
        return task.cont
        