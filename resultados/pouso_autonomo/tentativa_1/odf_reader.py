import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "lualatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

a = pd.read_csv('reward_curve.csv', sep='\t')
reward_avg = (a[['Unnamed: 6']]).to_numpy(dtype=str)
index = (a[['Unnamed: 0']]).to_numpy(dtype=np.float)

reward_corrected = []
for reward in reward_avg[:, 0]:
    reward_corrected.append(np.char.replace(reward,',','.'))

index = index[:,0]
reward_corrected = np.array(reward_corrected).astype(np.float)

plt.plot(index, reward_corrected)
plt.xlabel('Episódios de Treinamento')
plt.ylabel('Recompensa')
plt.grid()
# plt.show()

plt.savefig('recompensa.pgf') 