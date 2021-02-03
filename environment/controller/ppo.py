import sys
sys.path.append('/home/rafaelcostaf/mestrado/quadrotor_environment/')
from datetime import datetime, date

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import time

from environment.quadrotor_env import quad, plotter
from environment.controller.dl_auxiliary import dl_in_gen
from environment.controller.model import ActorCritic
from multiprocessing import Process, Queue, Pool



import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-s', action='store', dest = 'seed', help='random seed', type=int)
parser.add_argument('-N', action='store', dest = 'size', help='NN hidden size', type=int)
results = parser.parse_args()


"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL
NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR
DESCRIPTION:
    PPO deep learning training algorithm. 
"""
random_seed = int(results.seed)
network_size = int(results.size)
print('Tamanho N da Rede Neural: {:d}'.format(network_size))

seed = '_velocity_seed_'+str(network_size)+'0'+str(random_seed)
device = torch.device("cpu")
# device = torch.device("cuda:0")
# torch.set_num_threads(16)
PROCESS_TIME = time.time()

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]
    
    def convert_memory(self):
        self.actions = torch.stack(self.actions)
        self.states = torch.stack(self.states)
        self.logprobs = torch.stack(self.logprobs)
        self.rewards = torch.Tensor(self.rewards)
        self.is_terminals = torch.Tensor(self.is_terminals)
        self.values = torch.Tensor(self.values)
        
class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(network_size, state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(network_size, state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        try:
            self.policy.load_state_dict(torch.load('./PPO_continuous_drone'+seed+'.pth',map_location=device))
            self.policy_old.load_state_dict(torch.load('./PPO_continuous_old_drone'+seed+'.pth',map_location=device))
            print('Saved models loaded')
        except:
            print('New models generated')
            pass
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    
    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0
        lmbda = 0.99
        gmma = 0.99
        values = values.detach().cpu().numpy()
        for i in reversed(range(len(rewards))):
            if i == len(rewards):
                delta = rewards[i] - values[i]
            else:
                delta = rewards[i] + gmma * values[i + 1] * masks[i] - values[i]
            gae = delta + gmma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]

        return np.array(returns), (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        # rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.Tensor(memory.states).type(torch.float).to(device).detach()
        old_actions = torch.Tensor(memory.actions).type(torch.float).to(device).detach()
        old_logprobs = torch.Tensor(memory.logprobs).type(torch.float).to(device).detach().flatten()
        old_values = torch.Tensor(memory.values).type(torch.float).to(device).detach()
        rewards = np.array(memory.rewards).astype(float)
        # advantages = rewards - old_values.detach()[0] 
        
        rewards, advantages = self.get_advantages(old_values, np.logical_not(memory.is_terminals), rewards)
        advantages = torch.tensor(advantages).type(torch.float).to(device)
        rewards = torch.tensor(rewards).type(torch.float).to(device)
        # advantages = (advantages - advantages.mean())/(advantages.std()+1e-5)
        # Optimize policy for K epochs:
        # print(memory.rewards, rewards, memory.is_terminals)
        # print(old_states, old_actions, old_logprobs, old_values, advantages)
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            rd_idx = torch.randperm(np.shape(old_states)[0])
            old_states_sp = old_states[rd_idx]
            old_actions_sp = old_actions[rd_idx]
            old_logprobs_sp = old_logprobs[rd_idx]
            advantages_sp = advantages[rd_idx].flatten()
            rewards_sp = rewards[rd_idx]    
            
            # print(rd_idx.shape, old_states_sp.shape, old_actions_sp.shape, old_logprobs_sp.shape, advantages_sp.shape, rewards_sp.shape)
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_sp, old_actions_sp)
            # print(logprobs.type(), state_values.type(), dist_entropy.type())
            # print(old_states_sp.type(), old_actions_sp.type(), old_logprobs_sp.type(), advantages_sp.type(), rewards_sp.type())
            # advantages_sp = rewards_sp - state_values
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs.flatten() - old_logprobs_sp.detach().flatten())


            # Finding Surrogate Loss:
            # print(ratios.size(), logprobs.size(), old_logprobs_sp.size(), advantages_sp.size(), state_values.size(), rewards_sp.size())
            surr1 = ratios * advantages_sp
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_sp
            critic_loss = 0.5*self.MseLoss(state_values, rewards_sp)
            actor_loss = -torch.min(surr1, surr2)
            entropy_loss = - 0.01*dist_entropy.flatten()

            # print(critic_loss, actor_loss, entropy_loss)
            # print(actor_loss.size(), critic_loss.size(), entropy_loss.size())
            loss = actor_loss + critic_loss + entropy_loss
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

class worker_f():
    def __init__ (self, size):
        self.env = quad(time_int_step, max_timesteps, euler=0, direct_control=1, T=T)
        self.size = size
        self.memory = Memory()
        self.aux_dl = dl_in_gen(T, env.state_size, env.action_size)  
        self.aux_dl.reset()
        
    def work(self, ppo, eval_flag = False):
        self.time_step = 0
        while True:
            state, action = self.env.reset()
            t_since_last_plot = 0
            self.aux_dl.reset()
            self.reward_sum = 0
            for t in range(max_timesteps):        
                # Converts the quadrotor past states and actions to neural network input
                network_in = self.aux_dl.dl_input(state, action)        

                t_since_last_plot += 1
                self.time_step +=1
                
                # Running policy_old:
                if eval_flag:
                    action = ppo.policy.actor(torch.FloatTensor(network_in).to(device)).cpu().detach().numpy()  
                else:
                    # print('antes')
                    action = ppo.select_action(network_in, self.memory)
                    # print('depois')
                state, reward, done = self.env.step(action)
                action = np.array([action])
                
                # Saving reward and is_terminals:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                self.reward_sum += reward
                self.solved = self.env.solved
                # print(self.time_step)
                if done:
                    break
            if (self.time_step > self.size and done) or eval_flag:
                if not eval_flag:
                    self.memory.convert_memory()
                return self.memory, self.reward_sum, self.solved, self.time_step
            

def evaluate(env, agent, plotter, eval_steps=10):
    n_solved = 0
    rewards = 0
    time_steps = 0
    for i in range(int(eval_steps/N_WORKERS)):
        
        thrd_list = []             
        for i in range(N_WORKERS):
            thrd_list.append(p.apply_async(w_list[i].work, (ppo, True, )))
                
        for thread, worker in zip(thrd_list, w_list):    
            _, worker_reward_sum, worker_solved, worker_time_step, = thread.get()
            rewards += worker_reward_sum
            n_solved += worker_solved
            time_steps += worker_time_step
            
    time_mean = time_steps/eval_steps
    solved_mean = n_solved/eval_steps        
    reward_mean = rewards/eval_steps
    # plotter.plot()
    
    return reward_mean, time_mean, solved_mean
    
## HYPERPARAMETERS - CHANGE IF NECESSARY ##
lr = 0.0005
max_timesteps = 1000
action_std = 0.1
update_timestep = 5000
K_epochs = 10
T = 5
N_WORKERS = 2

## HYPERPAREMETERS - PROBABLY NOT NECESSARY TO CHANGE ##
action_dim = 4

log_interval = 5
eval_episodes = 10
max_episodes = 100000
max_trainings = 2000
time_int_step = 0.01
solved_reward = 700
eps_clip = 0.2
gamma = 0.99 
betas = (0.9, 0.999)
DEBUG = 0

# creating environment
env = quad(time_int_step, max_timesteps, euler=0, direct_control=1, T=T)
state_dim = 15*T
plot = plotter(env, True, False)

#creating reward logger
if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

# creating ppo trainer
memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
print(lr,betas)

# logging variables
t_since_last_plot = 0


time_step = 0
solved_avg = 0
training_count = 0
log = False
eval_on_mean = True

file_logger = open('eval_reward_log'+seed+'.txt', 'a')
file_logger.write('Gamma: ' + str(gamma)+'\t'+'Betas: ' + str(betas)+'\t'+'LR: '+str(lr)+'\t'+'Ep. Timstp: ' + str(max_timesteps)+'\t'+'Up. Timestep: ' + str(update_timestep)+'\t'+'Batch Epochs: ' + str(K_epochs)+'/t'+'Action Std: ' + str(action_std)+'\n')
file_logger.close()
        
# auxiliar deep learning input generator
aux_dl = dl_in_gen(T, env.state_size, env.action_size)  
aux_dl.reset()
time_list = []

w_list = []
env_list = []
thrd_list = []
for i in range(N_WORKERS):
    w_list.append(worker_f(int(update_timestep/N_WORKERS)))
    
p = Pool(N_WORKERS)  

# training loop
for i_episode in range(1, max_episodes+1):
    print('\rProgress: {:.2%}'.format(training_count/max_trainings), end='')
    
    thrd_list = []  
       
    for i in range(N_WORKERS):
        thrd_list.append(p.apply_async(w_list[i].work, (ppo, False)))
            
    for thread, worker in zip(thrd_list, w_list):    
        worker_memory, _, _, _, = thread.get()

        memory.states += worker_memory.states.tolist()
        memory.logprobs += worker_memory.logprobs.tolist()
        memory.rewards += worker_memory.rewards.tolist()
        memory.is_terminals += worker_memory.is_terminals.tolist()
        memory.values += worker_memory.values.tolist()
        memory.actions += worker_memory.actions.tolist()
    
    
    memory.values.append(torch.tensor([[0]]).to(device))
    time_init = time.time()
    ppo.update(memory)
    time_end = time.time()-time_init
    time_list.append(time_end)
    memory.clear_memory()
    time_step = 0
    training_count += 1
    log = True

    
    # save every x episodes
    if i_episode % log_interval == 0:
        torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format('drone'+seed))
        torch.save(ppo.policy_old.state_dict(), './PPO_continuous_old_{}.pth'.format('drone'+seed))
        
    # logging
    if training_count % log_interval == 0 and log:
        log = False
        reward_avg, time_avg, solved_avg = evaluate(env, ppo, plot, eval_episodes)


        print('\rEpisode {} \t Avg length: {} \t Avg reward: {:.2f} \t Solved: {:.2f}'.format(i_episode, time_avg, reward_avg, solved_avg))


        
        today = date.today()
        # dd/mm/YY
        d1 = today.strftime("%d/%m/%Y")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        file_logger = open('eval_reward_log'+seed+'.txt', 'a')
        file_logger.write(d1+'\t'+current_time+'\t'+str(training_count)+'\t'+str(i_episode)+'\t'+str(time.time()-PROCESS_TIME)+'\t'+str(reward_avg)+'\t'+str(solved_avg)+'\t'+str(time_avg)+'\n')
        file_logger.close()
        # stop training if avg_reward > solved_reward
        if training_count >= max_trainings:
            # reward_avg, time_avg, solved_avg = evaluate(env, ppo, plot, 200)
            # print('\rRe-evaluation \t Avg length: {} \t Avg reward: {:.2f} \t Solved: {:.2f}'.format(time_avg, reward_avg, solved_avg))
            # if solved_avg > 0.95 and reward_avg>solved_reward:
                print(np.average(np.array(time_list)))
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format('drone'+seed))
                torch.save(ppo.policy_old.state_dict(), './PPO_continuous_old_solved_{}.pth'.format('drone'+seed))
                break
                