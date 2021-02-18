import pandas as pd
import numpy as np
import matplotlib
import glob 

import matplotlib.pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.preamble':[
        '\DeclareUnicodeCharacter{2212}{-}']
})
matplotlib.rcParams['pgf.preamble'] = '\DeclareUnicodeCharacter{2212}{-}'

df_array = []

for file in glob.glob('*eval.csv'):
    df_array.append(pd.read_csv(file))
        
frame = pd.concat(df_array, axis=0, ignore_index = True)
dv_media = frame['Delta V'].mean()
tm_media = frame['Total Time'].where(frame['Solved']==1).mean()
sv_prop = frame['Solved'].mean()

print('Proporcao de Acertos: {:.2%} Delta V: {:.2f} Tempo: {:.2f}'.format(sv_prop, dv_media, tm_media))