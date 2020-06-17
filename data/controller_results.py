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

arq = glob.glob('./random_initial_state/*_true_state.npz')

for i, file in enumerate(arq):
    a = np.load(file)
    if i == 0:
        w = a[a.files[0]][0:3000]
        z = a[a.files[1]][0:3000]
        contr_error = w
        est_error = z
    else:
        w = a[a.files[0]][0:3000]
        z = a[a.files[1]][0:3000]
        contr_error = np.dstack((contr_error, w))
        est_error = np.dstack((contr_error, z))

av_contr_error = np.mean(abs(contr_error[:,0:5:2]), axis=(1,2))
var_contr_error = np.var(abs(contr_error[:,0:5:2]), axis=(1,2))


t = np.linspace(0, 999, 1000)*0.01


fig, ax1 = plt.subplots(1, 1, figsize=(7,7))

ax1.plot(t, av_contr_error, color='b')
# ax2.plot(t, av_contr_error[:,2], color='g')
# ax3.plot(t, av_contr_error[:,4], color='r')

# ax1.plot(t, av_contr_error_img, color='violet')
# ax2.plot(t, av_contr_error_img[:,2], color='cyan')
# ax3.plot(t, av_contr_error_img[:,4], color='gold')

# fig.suptitle('Average Quadrotor Position Over Time - Hover Flight')
# ax1.set_title('Average Axis Absolute Position Over Time')
# ax2.set_title('Y axis')
# ax3.set_title('Z axis')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Axis Average Absolute Position (m)')

ax1.fill_between(t, av_contr_error-var_contr_error, av_contr_error+var_contr_error, alpha=0.4, facecolor='b')
# ax2.fill_between(t, av_contr_error[:,2]-var_contr_error[:,2], av_contr_error[:,2]+var_contr_error[:,2], alpha=0.4, facecolor='g')
# ax3.fill_between(t, av_contr_error[:,4]-var_contr_error[:,4], av_contr_error[:,4]+var_contr_error[:,4], alpha=0.4, facecolor='r')


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
