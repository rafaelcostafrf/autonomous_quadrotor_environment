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
import pandas as pd
import uuid





import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-s', action='store', dest = 'seed', help='random seed', type=int)
parser.add_argument('-N', action='store', dest = 'size', help='NN hidden size', type=int)
parser.add_argument('-id', action='store', dest = 'id', help = 'Unique ID', type = str, default=None)
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
N_WORKERS = 2
max_trainings = 2000
random_seed = int(results.seed)*max_trainings*N_WORKERS
network_size = int(results.size)
print('Neural Network N size: {:d}'.format(network_size))

if results.id:
    un_id = results.id
else:
    un_id = uuid.uuid4().hex
print('Seed Number: '+str(random_seed))
print('Unique ID: '+str(un_id))    
seed = '_'+str(network_size)+'_'+str(random_seed)+'_'+un_id
device = torch.device("cpu")
# device = torch.device("cuda:0")
# torch.set_num_threads(16)
PROCESS_TIME = time.time()


header = ['LR', 'Ep_timesteps', 'Up_timesteps', 'Batch_Epochs', 'Eval Episodes', 'Action_std', 'Date', 'Hour', 'N_Training', 'T_seconds', 'Avg_reward', 'Solved Avg', 'Avg_length', 'Total Episodes', 'Total Timesteps', 'ETF']
try:
    dataframe = pd.read_csv('./training_log/log'+seed+'.csv')
    print('Dataframe from seed {:d} Loaded'.format(random_seed))
except:    
    print('Could not load dataframe, created a new one.')
    dataframe = pd.DataFrame(columns=header)

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
        
        self.policy = ActorCritic(network_size, state_dim, action_dim, action_std, FIXED_STD).double().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(network_size, state_dim, action_dim, action_std, FIXED_STD).double().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        try:
            self.policy.load_state_dict(torch.load('./untrained_networks/nn'+seed+'.pth',map_location=device))
            self.policy_old.load_state_dict(torch.load('./untrained_networks/nn_old'+seed+'.pth',map_location=device))
            print('Saved models loaded')
        except:
            print('New models generated')
            pass
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.DoubleTensor(state.reshape(1, -1)).to(device)
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
        old_states = torch.Tensor(memory.states).type(torch.double).to(device).detach()
        old_actions = torch.Tensor(memory.actions).type(torch.double).to(device).detach()
        old_logprobs = torch.Tensor(memory.logprobs).type(torch.double).to(device).detach()
        old_values = torch.Tensor(memory.values).type(torch.double).to(device).detach()
        rewards = np.array(memory.rewards)
        # advantages = rewards - old_values.detach()[0] 
        
        rewards, advantages = self.get_advantages(old_values, np.logical_not(memory.is_terminals), rewards)
        advantages = torch.tensor(advantages).type(torch.double).to(device)
        rewards = torch.tensor(rewards).type(torch.double).to(device)
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

            ratios = torch.exp(logprobs.sum(axis=2).flatten() - old_logprobs_sp.sum(axis=2).detach().flatten())
            


            # Finding Surrogate Loss:
            # print(ratios.size(), logprobs.mean(axis=2).size(), old_logprobs_sp.mean(axis=2).size(), advantages_sp.size(), state_values.size(), rewards_sp.size())
            surr1 = ratios * advantages_sp
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_sp
            critic_loss = 0.5*self.MseLoss(state_values, rewards_sp)
            actor_loss = -torch.min(surr1, surr2)
            entropy_loss = - 0.006*dist_entropy.sum(axis=2).flatten()

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
    def __init__ (self, size, n):
        
        self.env = quad(time_int_step, max_timesteps, training=True, euler=0, direct_control=1, T=T)
        self.n = n
        
        
        self.size = size
        self.memory = Memory()
        self.aux_dl = dl_in_gen(T, self.env.state_size, self.env.action_size)  
        self.aux_dl.reset()
        
    def work(self, ppo, random_seed, eval_flag = False):
        local_random_seed = random_seed+self.n
        # print(local_random_seed)
        np.random.seed(random_seed+self.n)
        torch.manual_seed(random_seed+self.n)
        self.env.seed(random_seed+self.n)
        
        self.time_step = 0
        self.episodes = 0
        self.memory.clear_memory()
        while True:
            state, action = self.env.reset()
            t_since_last_plot = 0
            # self.aux_dl.reset()
            self.reward_sum = 0
            for t in range(max_timesteps):        
                # Converts the quadrotor past states and actions to neural network input
                network_in = self.aux_dl.dl_input(state, action)        

                t_since_last_plot += 1
                self.time_step +=1
                
                # Running policy_old:
                if eval_flag:
                    action = ppo.policy.actor(torch.DoubleTensor(network_in).to(device)).cpu().detach().numpy()  
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

                if done:
                    self.episodes += 1
                    break
                if t == max_timesteps - 1:
                    self.episodes += 1
            if (self.time_step > self.size and done) or eval_flag:
                if not eval_flag:
                    self.memory.convert_memory()
                return self.memory, self.reward_sum, self.solved, self.time_step, self.episodes
            

def evaluate(agent, eval_steps=10):
    n_solved = 0
    rewards = 0
    time_steps = 0
    for i in range(int(eval_steps/N_WORKERS)):
        
        thrd_list = []            
        evaluate_random_seed = random_seed+evaluate_count*eval_steps+N_WORKERS*i
        for j in range(N_WORKERS):
            thrd_list.append(p.apply_async(w_list[j].work, (ppo, evaluate_random_seed, True)))
                
        for thread, worker in zip(thrd_list, w_list):    
            _, worker_reward_sum, worker_solved, worker_time_step, _ = thread.get()
            rewards += worker_reward_sum
            n_solved += worker_solved
            time_steps += worker_time_step
            
    total_ep = int(eval_steps/N_WORKERS)*N_WORKERS
    time_mean = time_steps/total_ep
    solved_mean = n_solved/total_ep
    reward_mean = rewards/total_ep

    
    return reward_mean, time_mean, solved_mean
    
## HYPERPARAMETERS - CHANGE IF NECESSARY ##
lr = 0.0005
max_timesteps = 1000
action_std = 0.1
FIXED_STD = True
update_timestep = 5000
K_epochs = 10
T = 5


## HYPERPAREMETERS - PROBABLY NOT NECESSARY TO CHANGE ##
action_dim = 4

log_interval = 5
eval_episodes = 40
max_episodes = 100000

time_int_step = 0.01
solved_reward = 700
eps_clip = 0.2
gamma = 0.99 
betas = (0.9, 0.999)
DEBUG = 0

# creating environment
state_dim = 15*T


if random_seed:
    print("Random Seed: {}".format(random_seed))


# creating ppo trainer
memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
print(lr,betas)

# logging variables
t_since_last_plot = 0


time_step = 0
solved_avg = 0
training_count = 0
evaluate_count = 0

total_episodes = 0 
total_timesteps = 0
log = False
eval_on_mean = True


        
# auxiliar deep learning input generator
time_list = []

w_list = []
env_list = []
thrd_list = []
for i in range(N_WORKERS):
    w_list.append(worker_f(int(update_timestep/N_WORKERS), i))

p = Pool(N_WORKERS)  

# training loop
for i_episode in range(1, max_episodes+1):
    print('Progress: {:.2%}'.format(training_count/max_trainings), end='          \r')
    
    thrd_list = []  
    episode_random_seed = random_seed+N_WORKERS*training_count
    
    for i in range(N_WORKERS):
        thrd_list.append(p.apply_async(w_list[i].work, (ppo, episode_random_seed, False)))
            
    for thread, worker in zip(thrd_list, w_list):    
        worker_memory, _, _, worker_i, worker_episodes = thread.get()

        total_episodes += worker_episodes 
        total_timesteps += worker_i 
        
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
        torch.save(ppo.policy.state_dict(), './untrained_networks/nn{}.pth'.format(seed))
        torch.save(ppo.policy_old.state_dict(), './untrained_networks/nn_old{}.pth'.format(seed))
        
    # logging
    if training_count % log_interval == 0 and log:
        log = False
        reward_avg, time_avg, solved_avg = evaluate(ppo, eval_episodes)
        evaluate_count += 1

        print('Episode {:6d} Avg length: {:8.2f} Avg reward: {:8.2f} Solved: {:7.2%} Std: {:8.4f}'.format(i_episode, time_avg, reward_avg, solved_avg, ppo.policy.std.detach().cpu().numpy()))

        today = date.today()
        # dd/mm/YY
        d1 = today.strftime("%d/%m/%Y")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        file_logger = pd.Series([lr, max_timesteps, update_timestep, K_epochs, eval_episodes, ppo.policy.std.detach().cpu().numpy(), d1, current_time, training_count, time.time()-PROCESS_TIME, reward_avg, solved_avg, time_avg, total_episodes, total_timesteps, ((time.time()-PROCESS_TIME)/training_count*max_trainings-(time.time()-PROCESS_TIME))/3600], index = header)
        dataframe = dataframe.append(file_logger, ignore_index = True)
        with open('./training_log/log'+seed+'.csv', 'w') as f:
            dataframe.to_csv(f, index=False)
        
        # stop training if avg_reward > solved_reward
        if training_count >= max_trainings:
            # reward_avg, time_avg, solved_avg = evaluate(env, ppo, plot, 200)
            # print('\rRe-evaluation \t Avg length: {} \t Avg reward: {:.2f} \t Solved: {:.2f}'.format(time_avg, reward_avg, solved_avg))
            # if solved_avg > 0.95 and reward_avg>solved_reward:
                print(np.average(np.array(time_list)))
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), './solved/nn_solved{}.pth'.format(seed))
                torch.save(ppo.policy_old.state_dict(), './solved/nn_old_solved{}.pth'.format(seed))
                break
                