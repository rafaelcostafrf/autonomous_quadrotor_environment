import pandas as pd
import numpy as np
import glob 
import os 
import matplotlib.pyplot as plt


array_64 = []
array_128 = []
array_256 = []
for file in glob.glob('*.csv'):
    if int(file[4:7]) == 64:
        array_64.append(pd.read_csv(file))
    elif int(file[4:7]) == 128:
        array_128.append(pd.read_csv(file))
    else:
        array_256.append(pd.read_csv(file))

reward_array_64 = np.zeros([400, 4])
for i, df in enumerate(array_64):
    reward = df['Avg_reward'].to_numpy()
    reward_array_64[:, i] = reward
    
reward_array_128 = np.zeros([400, 4])
for i, df in enumerate(array_128):
    reward = df['Avg_reward'].to_numpy()
    reward_array_128[:, i] = reward
    
reward_array_256 = np.zeros([400, 4])
for i, df in enumerate(array_256):
    reward = df['Avg_reward'].to_numpy()
    reward_array_256[:, i] = reward
    
mean_reward_64 = np.mean(reward_array_64, axis = 1)
mean_reward_128 = np.mean(reward_array_128, axis = 1)
mean_reward_256 = np.mean(reward_array_256, axis = 1)

std_reward_64 = np.std(reward_array_64, axis = 1)
std_reward_128 = np.std(reward_array_128, axis = 1)
std_reward_256 = np.std(reward_array_256, axis = 1)


fig, axs = plt.subplots(3, figsize = (8, 8))
x = np.arange(400)

axs[0].plot(x, mean_reward_64)
axs[0].fill_between(x, mean_reward_64-std_reward_64, mean_reward_64+std_reward_64, alpha = 0.5)
axs[1].plot(mean_reward_128)
axs[1].fill_between(x, mean_reward_128-std_reward_128, mean_reward_128+std_reward_128, alpha = 0.5)
axs[2].plot(mean_reward_256)
axs[2].fill_between(x, mean_reward_256-std_reward_256, mean_reward_256+std_reward_256, alpha = 0.5)

plt.show()