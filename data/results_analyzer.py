import numpy as np
from matplotlib import pyplot as plt
import glob
plt.close('all')
"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    3D enviornment results analyzer
"""

t_total = 3000

def padding(z, length):
    if len(z) < length:
        pad = np.array([z[-1]])
        for i in range(length-len(z)):
            z = np.concatenate((z, pad), axis=0)
    return z

def read_data(files):
    for i, file in enumerate(files):
        a = np.load(file)
        if i == 0:
            w = a[a.files[0]][0:t_total]
            contr_error = padding(w, t_total)
            
            z = a[a.files[1]][0:t_total]
            est_error = padding(z, t_total)

        else:
            w = a[a.files[0]][0:t_total]
            w = padding(w, t_total)
            
            z = a[a.files[1]][0:t_total]
            z = padding(z, t_total)
            
            contr_error = np.dstack((contr_error, w))
            est_error = np.dstack((est_error, z))
    return contr_error, est_error    
    
def abs_avg(data):
    avg = []
    var = []
    for data_i in data:
        avg.append(np.mean(abs(data_i), axis=2))
        var.append(np.var(abs(data_i), axis=2))
    return avg, var

def fig_plot(t, avg, var, avg_2, var_2, name):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,6))
    fig.canvas.set_window_title(name)
    
    ax1.plot(t, avg[:,0], color='b')
    ax2.plot(t, avg[:,1], color='g')
    ax3.plot(t, avg[:,2], color='r')
    
    ax1.plot(t, avg_2[:,0], color='violet')
    ax2.plot(t, avg_2[:,1], color='cyan')
    ax3.plot(t, avg_2[:,2], color='gold')
    
    ax1.set_title('X axis')
    ax2.set_title('Y axis')
    ax3.set_title('Z axis')
    ax3.set_xlabel('Time (s)')
    ax2.set_ylabel('Absolute Position (m)')
    
    ax1.fill_between(t, avg[:,0]-var[:,0], avg[:,0]+var[:,0], alpha=0.4, facecolor='b')
    ax2.fill_between(t, avg[:,1]-var[:,1], avg[:,1]+var[:,1], alpha=0.4, facecolor='g')
    ax3.fill_between(t, avg[:,2]-var[:,2], avg[:,2]+var[:,2], alpha=0.4, facecolor='r')
    
    ax1.fill_between(t, avg_2[:,0]-var_2[:,0], avg_2[:,0]+var_2[:,0], alpha=0.4, facecolor='violet')
    ax2.fill_between(t, avg_2[:,1]-var_2[:,1], avg_2[:,1]+var_2[:,1], alpha=0.4, facecolor='cyan')
    ax3.fill_between(t, avg_2[:,2]-var_2[:,2], avg_2[:,2]+var_2[:,2], alpha=0.4, facecolor='gold')
    
    ax1.legend(['MEMS', 'Hybrid'])
    ax2.legend(['MEMS', 'Hybrid'])
    ax3.legend(['MEMS', 'Hybrid'])
    
    ax1.grid(True, alpha=0.25)
    ax2.grid(True, alpha=0.25)
    ax3.grid(True, alpha=0.25)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    plt.tight_layout()
    
folders = ('hover', 'random_initial_state')

for folder in folders:
    folder_path = './'+folder+'/'
    arq = glob.glob(folder_path+'*_MEMS.npz')
    arq_img = glob.glob(folder_path+'*_Hybrid.npz')
    
    error_mems = read_data(arq)
    error_hyb = read_data(arq_img)
    
    avg_mems, var_mems = abs_avg(error_mems)
    avg_hyb, var_hyb = abs_avg(error_hyb)
    
    t = np.linspace(0, t_total-1, t_total)*0.01
    
    names = (folder + ' Control', folder + ' Estimation')
    for avg, var, avg_2, var_2, name in zip(avg_mems, var_mems, avg_hyb, var_hyb, names):
        fig_plot(t, avg, var, avg_2, var_2, name)

    print('\n------ ' + folder + ' Estimation Error ------ \n')
    pos_err = np.mean(abs(error_mems[1][:,0:3,:]))
    pos_var = np.var(abs(error_mems[1][:,0:3,:]))
    quat_err = np.mean(abs(error_mems[1][:,-4:,:]))
    quat_var = np.var(abs(error_mems[1][:,-4:,:]))
    print('MEMS')
    print(f'Position: {pos_err:.6f}+-{pos_var:.6f}    Quaternion: {quat_err:.6f}+-{quat_var:.6f}')
    
    pos_err_img = np.mean(abs(error_hyb[1][:,0:3,:]))
    pos_var_img = np.var(abs(error_hyb[1][:,0:3,:]))
    quat_err_img = np.mean(abs(error_hyb[1][:,-4:,:]))
    quat_var_img = np.var(abs(error_hyb[1][:,-4:,:]))
    print('Hybrid')
    print(f'Position : {pos_err_img:.6f}+-{pos_var_img:.6f}    Quaternion: {quat_err_img:.6f}+-{quat_var_img:.6f}')