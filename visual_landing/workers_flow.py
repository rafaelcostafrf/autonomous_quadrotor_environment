import torch
import time

from visual_landing.quad_worker import quad_worker
from visual_landing.ppo_worker import ppo_worker

# LANDING SETUP
from visual_landing.ppo_aux import PPO
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_WORKERS = 1
BATCH_SIZE = 300

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
    
    def __init__(self, render, cv_cam):
        
        self.MEMORY = Memory()
        self.render = render
        self.cv_cam = cv_cam
        self.ldg_policy = PPO(0, 3)
        
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
        self.workers = []
        for i in range(N_WORKERS):
            self.workers.append(quad_worker(self.render))
            self.render.taskMgr.add(self.workers[i].step, 'quad_worker'+str(i), taskChain = str(i))            
        
        self.ppo_worker = ppo_worker(self.render, self.workers, self.cv_cam, self.ldg_policy)     
        self.render.taskMgr.add(self.ppo_worker.wait_until_ready, 'ppo_worker'+str(i) , taskChain = 'ppo')
    

    def episode_done_check(self, task):
        done = True

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
                
                
                if len(self.MEMORY.rewards) >= BATCH_SIZE:
                    f = open('child_processes.txt', 'r')
                    lines = f.readlines()
                    f.close()
                    for line in lines:
                        while True:
                            s = open('./child_data/'+line.splitlines()[0]+'.txt', 'r')                            
                            a = int(s.read())
                            s.close()
                            if a == 1:
                                child_name = './child_data/'+line.splitlines()[0]
                               
                                actions_temp = torch.load(child_name+'actions.tch')
                                for i, action in enumerate(actions_temp):
                                    actions_temp[i] = action.to(device).detach()
                                    
                                states_temp = torch.load(child_name+'states.tch')
                                for i, state in enumerate(states_temp):
                                    states_temp[i] = state.to(device).detach()
                                    
                                logprobs_temp = torch.load(child_name+'logprobs.tch')
                                for i, logprob in enumerate(logprobs_temp):
                                    logprobs_temp[i] = logprob.to(device).detach()
                                    
                                rewards_temp = torch.load(child_name+'rewards.tch')

                               	is_terminals_temp = torch.load(child_name+'is_terminals.tch')


                                
                                self.MEMORY.actions.extend(actions_temp)
                                self.MEMORY.states.extend(states_temp)
                                self.MEMORY.logprobs.extend(logprobs_temp)
                                self.MEMORY.rewards.extend(rewards_temp)
                                self.MEMORY.is_terminals.extend(is_terminals_temp)
                                
                                del actions_temp
                                del states_temp
                                del logprobs_temp
                                del rewards_temp
                                del is_terminals_temp
                                break
                            else:                            
                                time.sleep(3)
                    print(len(self.MEMORY.actions))
                    self.ldg_policy.update(self.MEMORY)                    
                    for line in lines:
                        s = open('./child_data/'+line.splitlines()[0]+'.txt', 'w')    
                        s.write(str(0))
                        s.close()
                    self.MEMORY.clear_memory()
                    torch.cuda.empty_cache()
            for worker in self.workers:
                worker.memory.clear_memory()
            self.reset_workers()
        return task.cont
        