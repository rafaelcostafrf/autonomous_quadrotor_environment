import torch
import time
import sys
import numpy as np

from visual_landing.quad_worker import quad_worker
from visual_landing.ppo_worker_v2 import ppo_worker
from visual_landing.memory import Memory

# LANDING SETUP
from visual_landing.ppo_aux_v2 import PPO
device = torch.device("cpu")
from visual_landing.memory_leak import debug_gpu


BATCH_SIZE = 350

class work_flow():
    
    def __init__(self, render, cv_cam, child_number=None, child = False):
        
        self.child = child
        if child:
            self.child_number = child_number
        self.MEMORY = Memory()
        self.render = render
        self.cv_cam = cv_cam
        self.ldg_policy = PPO(3, 3)
        
        self.render.taskMgr.setupTaskChain('worker', numThreads = 1, tickClock = None,
                                   threadPriority = None, frameBudget = None,
                                   frameSync = None, timeslicePriority = None)

        self.render.taskMgr.setupTaskChain('ppo', numThreads = 1, tickClock = None,
                               threadPriority = None, frameBudget = None,
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

        
        if self.child:
            while True:
                f = open('./child_data/'+str(self.child_number)+'.txt', 'r')
                if int(f.read()) == 0:
                    break
                else:
                    time.sleep(3)            
            
            self.worker = quad_worker(self.render)
            self.render.taskMgr.add(self.worker.step, 'quad_worker', taskChain = 'worker')  
            try:
                self.ldg_policy.policy.load_state_dict(torch.load('./PPO_landing.pth', map_location=device))
                self.ldg_policy.policy_old.load_state_dict(torch.load('./PPO_landing_old.pth', map_location=device))
            except:
                print("Could Not Load Father Landing Policy")
                sys.exit()
            self.ppo_worker = ppo_worker(self.render, self.worker, self.cv_cam, self.ldg_policy, child = True)     
            self.render.taskMgr.add(self.ppo_worker.wait_until_ready, 'ppo_worker', taskChain = 'ppo')
        
        else:
            self.worker = quad_worker(self.render)
            self.render.taskMgr.add(self.worker.step, 'quad_worker', taskChain = 'worker')            
            
            self.ppo_worker = ppo_worker(self.render, self.worker, self.cv_cam, self.ldg_policy)     
            self.render.taskMgr.add(self.ppo_worker.wait_until_ready, 'ppo_worker', taskChain = 'ppo')
    
    def child_save_data(self):        
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
        self.worker.memory.clear_memory()    


    def mother_train(self):
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
                        
                    states_temp = torch.load(child_name+'states.tch')
                        
                    logprobs_temp = torch.load(child_name+'logprobs.tch')
                     
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
        print('train')
        print(len(self.MEMORY.states))
        debug_gpu()
        self.ldg_policy.update(self.MEMORY)                    
        debug_gpu()
        for line in lines:
            s = open('./child_data/'+line.splitlines()[0]+'.txt', 'w')    
            s.write(str(0))
            s.close()
        self.MEMORY.clear_memory()
        torch.cuda.empty_cache()
        self.worker.memory.clear_memory()
        

    def episode_done_check(self, task):
       if self.worker.visual_done or len(self.worker.memory.states)+len(self.MEMORY.states) > BATCH_SIZE:
           self.MEMORY.actions.extend(self.worker.memory.actions)
           self.MEMORY.states.extend(self.worker.memory.states)
           self.MEMORY.logprobs.extend(self.worker.memory.logprobs)
           self.MEMORY.rewards.extend(self.worker.memory.rewards)
           self.MEMORY.is_terminals.extend(self.worker.memory.is_terminals)
           self.reset_workers()
       batch_size_percentage = len(self.MEMORY.rewards)/BATCH_SIZE   
       print('\rEpisode Completion: {:.2%}'.format(batch_size_percentage),end='         ')
       if len(self.MEMORY.rewards)>=BATCH_SIZE:
            if self.child:
                self.child_save_data() 
            else:
                self.mother_train()
            self.worker.memory.clear_memory()
            self.reset_workers()
            self.MEMORY.clear_memory()
            
       return task.cont
        