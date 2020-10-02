import numpy as np
import pandas as pd


def metrics_calculator(y, target):
    axis = [0, 2, 4]
    over = [0, 0, 0]
    rise_time = [0, 0, 0]
    settling_time = [0, 0, 0]
    ss_error = [0, 0, 0]
    for j, i in enumerate(axis):
        over[j] = np.max(np.abs(y[i, :]))
        try:
            rise_time[j] = (np.abs(np.where(y[i, :])-target[j])<0.05)[0][0]*0.01
        except:
            rise_time[j] = 0
        try:
            settling_time[j] = 50.01-np.where(np.flip(np.abs(y[i, :]-target[j]))>0.05*abs(target[j]))[0][0]*0.01
        except:
            settling_time[j] = 0
        ss_error[j] = y[i, -1]-target[j]
    return over, rise_time, settling_time, ss_error
    
    
def response_analyzer(y, target, control_effort, abs_error, env_max_steps):
    over, rise_time, settling_time, ss_error = metrics_calculator(y, target)
    a = [(control_effort/env_max_steps), (abs_error/env_max_steps), 
         over[0], over[1], over[2], 
         rise_time[0], rise_time[1], rise_time[2],
         settling_time[0], settling_time[1], settling_time[2],
         ss_error[0], ss_error[1], ss_error[2]]       
    
    S = pd.Series(a)
    S.index = ['CE', 'EOT',
              'Over X', 'Over Y', 'Over Z',
              'Rise X', 'Rise Y', 'Rise Z',
              'Set X', 'Set Y', 'Set Z',
              'SS X', 'SS Y', 'SS Z',]     
    return S

        