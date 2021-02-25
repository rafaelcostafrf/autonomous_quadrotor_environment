import numpy as np 
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

files_array = ['lqr_log.npy', 'lqr_log_not_clipped.npy', 'pid_log.npy', 'pid_log_not_clipped.npy', 'rl_log.npy']
header = ['Settling Time', 'Overshoot', 'Accum. Error', 'SS error', 'Control Effort', 'Max Control Effort', 'Succes Proportion']
df = pd.DataFrame(columns = header)
M, G, T2WR = 1.03, 9.82, 2

for file in files_array:
    a = np.load(file)
    
    succ = np.zeros(a.shape[0])
    epp = np.zeros(a.shape[0])
    efc = np.zeros(a.shape[0])
    efcmax = np.zeros(a.shape[0])
    ts = np.zeros(a.shape[0])
    ov = np.zeros(a.shape[0])
    ev = np.zeros(a.shape[0])
    
    def success(episode):
        if np.linalg.norm(episode[-1, 0:4]) < 0.05:
            return True
        else:
            return False
    
    
    def ts_calculator(episode):
        ts = None
        out = True
        i = 0
        last_norm = True
        norm = np.linalg.norm(episode[:, 0:3], axis = 1) < 0.05
        for norm_i in reversed(norm):            
            if norm_i == False and last_norm == True:
                ts = (a.shape[1]-i)*0.01
            i += 1
            last_norm = norm_i
        return ts   
    
    def ov_calculator(episode):
        ov = [0, 0, 0]
        negative = [False, False, False]
        negative_old = [False, False, False]
        passou = [0, 0, 0]
        for j, t in enumerate(episode):
            for i, ax in enumerate(t[0:3]):
                negative[i] = False if ax >= 0 else True
                if (negative[i] != negative_old[i]) and passou[i] == 0:
                    passou[i] = j
            negative_old = negative.copy()

        for i in range(3):   
            if passou[i] > 0:
                ov[i] = np.max(np.abs(episode[passou[i]:, i]))
        return np.mean(ov)
        
        
    for i, episode in enumerate(a):
        if success(episode):
            succ[i] = 1
            episode[:, -4:] = (episode[:, -4:] + 1)*M*G*T2WR/8
            ev[i] = np.mean(np.linalg.norm(episode[:, 0:3], axis = 1))
            epp[i] = np.linalg.norm(episode[-1, 0:3])
            efc[i] = np.mean(np.sum(np.abs(episode[:, -4:]), axis = 1))
            ts[i] = ts_calculator(episode)
            ov[i] = ov_calculator(episode)
            efcmax[i] = np.max(np.sum(np.abs(episode[:, -4:]), axis = 1))
            
    sa_mean = np.mean(succ)
    mask = np.logical_not(succ)
    pe_mean = np.mean(np.ma.masked_array(epp, mask = mask))
    ce_mean = np.mean(np.ma.masked_array(efc, mask = mask))
    cemax_mean = np.mean(np.ma.masked_array(efcmax, mask = mask))
    st_mean = np.mean(np.ma.masked_array(ts, mask = mask))
    ov_mean = np.mean(np.ma.masked_array(ov, mask = mask))
    ev_mean = np.mean(np.ma.masked_array(ev, mask = mask))
    data = (st_mean, ov_mean, ev_mean, pe_mean, ce_mean, cemax_mean, sa_mean)
    data = pd.Series(data, name = file, index = header)
    df = df.append(data)
    print('File: {} Success Average: {:.3%} P. Error Avg: {:.3e} Control Effort Avg: {:.3e} Settling Time Avg: {:.3e} Overshoot Mean: {:.3e} Total Error: {:.3e}'.format(file, sa_mean, pe_mean, ce_mean, st_mean, ov_mean, ev_mean))
print(df.transpose())