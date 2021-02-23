import numpy as np 

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.preamble':[
        '\DeclareUnicodeCharacter{2212}{-}']
})

import matplotlib.pyplot as plt

files_array = ['lqr_log_same_start.npy', 'lqr_log_same_start_not_clipped.npy', 'pid_log_same_start.npy', 'pid_log_same_start_not_clipped.npy', 'rl_log_same_start.npy']
Names = ['LQR', 'LQR SL', 'PID', 'PID SL', 'RL']
Axis_names = ['$\dot X$', '$\dot Y$', '$\dot Z$']
Action_names = ['$u_1$', '$u_2$', '$u_3$','$u_4$']
fig_array = []
ax_array = []

for (k, file), name in zip(enumerate(files_array), Names):
    a = np.load(file)
    if k == 0:
        for j in range(a.shape[0]):
            fig, ax = plt.subplots(5, 2, figsize = (7, 7*1.414), dpi=300)

            fig_array.append(fig)
            ax_array.append(ax)

    for i, episode in enumerate(a):
        vel = episode[:, 0:3]
        ang = episode[:, 3:6]
        action = episode[:, -4:]
        t = np.arange(0, a.shape[1])*0.01
        [x, y, z] = ax_array[i][k][0].plot(t, vel, linewidth = 0.5)
        [u1, u2, u3, u4] = ax_array[i][k][1].plot(t, action, linestyle = '--', linewidth = 0.5)
        if k == 0:
            ax_array[i][k][0].legend([x, y, z], Axis_names, loc = 1)
            ax_array[i][k][1].legend([u1, u2, u3, u4], Action_names, loc = 1)
        ax_array[i][k][1].title.set_text(name)
        ax_array[i][k][1].grid(True)
        ax_array[i][k][0].title.set_text(name)
        ax_array[i][k][0].grid(True)
        if k != 4:
            plt.setp(ax_array[i][k][0].get_xticklabels(), visible=False)
            plt.setp(ax_array[i][k][1].get_xticklabels(), visible=False)

for i, fig in enumerate(fig_array):    
    fig.text(0.5, 0.08, 'Tempo (s)', ha='center')
    fig.text(0.06, 0.5, 'Velocidade (m/s)', va='center', rotation='vertical')
    fig.savefig(str(i)+'_same_start.pgf',bbox_inches='tight')
    fig.savefig(str(i)+'_same_start.png',bbox_inches='tight')
