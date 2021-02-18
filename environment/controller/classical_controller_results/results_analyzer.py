import numpy as np 
files_array = ['lqr_log.npy', 'pid_log.npy', 'rl_log.npy']

for file in files_array:
    a = np.load(file)
    
    succ = np.zeros(a.shape[0])
    epp = np.zeros(a.shape[0])
    efc = np.zeros(a.shape[0])
    ts = np.zeros(a.shape[0])
    ov = np.zeros(a.shape[0])
    
    
    def success(episode):
        if np.linalg.norm(episode[-1, 0:4]) < 0.05:
            return True
        else:
            return False
    
    
    def ts_calculator(episode):
        ts = None
        for i, t in enumerate(episode):
            if np.linalg.norm(t[0:4]) >= 0.05:
                out = True
            if np.linalg.norm(t[0:4]) < 0.05 and out:
                out = False
                ts = i*0.01
        return ts   
    
    def ov_calculator(episode):
        ov = [None, None, None]
        negative = [False, False, False]
        negative_old = [False, False, False]
        passou = [0, 0, 0]
        for j, t in enumerate(episode):
            for i, ax in enumerate(t[0:3]):
                negative[i] = False if ax >= 0 else True
                if (negative[i] != negative_old[i]):
                    passou[i] = j
            negative_old = negative

        for i in range(3):    
            ov[i] = np.max(episode[passou[i]:, i])            
        return np.mean(ov)
        
        
    for i, episode in enumerate(a):
        if success(episode):
            succ[i] = 1
            epp[i] = np.linalg.norm(episode[-1, 0:3])
            efc[i] = np.mean(np.sum(np.abs(episode[:, -4:]), axis = 1))
            ts[i] = ts_calculator(episode)
            ov[i] = ov_calculator(episode)
            
    sa_mean = np.mean(succ)
    mask = np.logical_not(succ)
    pe_mean = np.mean(np.ma.masked_array(epp, mask = mask))
    ce_mean = np.mean(np.ma.masked_array(efc, mask = mask))
    st_mean = np.mean(np.ma.masked_array(ts, mask = mask))
    ov_mean = np.mean(np.ma.masked_array(ov, mask = mask))
    print('File: {} Success Average: {:.3%} P. Error Avg: {:.3e} Control Effort Avg: {:.3e} Settling Time Avg: {:.3e} Overshoot Mean: {:.3e}'.format(file, sa_mean, pe_mean, ce_mean, st_mean, ov_mean))