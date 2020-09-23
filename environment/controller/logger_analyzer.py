import numpy as np 
import glob
import matplotlib.pyplot as plt

path_files = glob.glob('./results/seeds/velocity_training/eval_reward_log_velocity_seed_*.txt')

global_episodes = np.zeros([len(path_files),1000])
global_rewards = np.zeros([len(path_files),1000])
global_ep_time = np.zeros([len(path_files),1000])
global_time_delta = np.zeros([len(path_files),1000])
global_last_ep = np.zeros([len(path_files)])
global_ep_time_mean = np.zeros([len(path_files)])
global_ep_time_mean_sol = np.zeros([len(path_files)])
min_ep_len = 1000
for n_file, file in enumerate(path_files):

    with open(file) as f:
        array = []
        for line in f:
            array.append([str(x) for x in line.split()])
    
    episode = []
    time = []
    reward = []
    ep_time = []    
    
    for i, line in enumerate(array):
        _, _, e, t, r, e_t = tuple(line)
        episode.append(float(e))
        time.append(float(t))
        reward.append(float(r))
        ep_time.append(float(e_t))
    
    episode = np.array(episode)
    time = np.array(time)
    reward = np.array(reward)
    ep_time = np.array(ep_time)
    time_delta = np.array([])
    
    for time_ant, time_i in zip(time[:-1],time[1::]):
        time_delta = np.append(time_delta, [time_i-time_ant])
        
    ep_len = len(episode)
    
    min_ep_len = ep_len if n_file==0 or ep_len<min_ep_len else min_ep_len
    
    global_episodes[n_file,:len(episode)] = episode
    global_rewards[n_file,:len(reward)] = reward
    global_ep_time[n_file,:len(ep_time)] = ep_time
    global_time_delta[n_file,:len(time_delta)] = time_delta
    sol_index = np.where(reward>660)[0][0]
    global_last_ep[n_file] = episode[sol_index]
    global_ep_time_mean[n_file] = np.mean(ep_time)
    global_ep_time_mean_sol[n_file] = np.mean(ep_time[0:sol_index])
    
rew_mean = np.mean(global_rewards[:,0:min_ep_len],axis=0)
rew_std = np.sqrt(np.var(global_rewards[:,0:min_ep_len],axis=0))
tim_mean = np.mean(global_time_delta[:,0:min_ep_len],axis=0)
tim_std = np.sqrt(np.var(global_time_delta[:,0:min_ep_len],axis=0))
ep_tim_mean = np.mean(global_ep_time[:,0:min_ep_len],axis=0)*0.01
ep_tim_std = np.sqrt(np.var(global_ep_time[:,0:min_ep_len],axis=0))*0.01


fig, ax = plt.subplots()
fig.canvas.set_window_title('Average Reward')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.grid(True)
ax.plot(global_episodes[0, 0:min_ep_len], rew_mean)
ax.fill_between(global_episodes[0, 0:min_ep_len], rew_mean-rew_std, rew_mean+rew_std, alpha=0.2)
plt.show()

fig, ax = plt.subplots()
fig.canvas.set_window_title('Average Iteration Time')
ax.set_xlabel('Episode')
ax.set_ylabel('time (s)')
ax.grid(True)
ax.plot(global_episodes[0, 0:min_ep_len], tim_mean)
ax.fill_between(global_episodes[0, 0:min_ep_len], tim_mean-tim_std, tim_mean+tim_std, alpha=0.2)
plt.show()

fig, ax = plt.subplots()
fig.canvas.set_window_title('Average Episode Time')
ax.set_xlabel('Episode')
ax.set_ylabel('time (s)')
ax.grid(True)
ax.plot(global_episodes[0, 0:min_ep_len], ep_tim_mean)
ax.fill_between(global_episodes[0, 0:min_ep_len], ep_tim_mean-ep_tim_std, ep_tim_mean+ep_tim_std, alpha=0.2)
plt.show()

var_last_ep = np.sqrt(np.var(global_last_ep))
var_ep_time = np.sqrt(np.var(global_ep_time_mean))
var_flight_time = np.sqrt((np.mean(global_last_ep)*var_ep_time)**2 + (np.mean(global_ep_time_mean)*var_last_ep)**2)

print('Average Ep Until Solution: ' + str(np.mean(global_last_ep))+' ± '+ str(var_last_ep)+' episodes')
print('Average Ep Time: ' + str(np.mean(global_ep_time_mean)) + ' ± '+ str(var_ep_time)+' steps')
print('Total Flight Time Until Solution: ' + str(np.mean(global_ep_time_mean_sol)*np.mean(global_last_ep)*0.01/60/60)+' ± '+ str(var_flight_time*0.01/60/60)+' hours')