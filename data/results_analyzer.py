import numpy as np
from matplotlib import pyplot as plt

"""
MECHANICAL ENGINEERING POST-GRADUATE PROGRAM
UNIVERSIDADE FEDERAL DO ABC - SANTO ANDRÉ, BRASIL

NOME: RAFAEL COSTA FERNANDES
RA: 21201920754
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIPTION:
    3D enviornment results analyzer
"""

a = np.load('random_img_det_results.npz')
b = np.load('random_accel_results.npz')


w = a[a.files[0]]
w = (w, b[b.files[0]])
for v in w:
    print('Posição')
    print('Média: %.4e +- Var: %.4e' %(np.mean(abs(v[:,0:3])), np.var(abs(v[:,0:3]))))
    print('Quaternion')
    print('Média: %.4e +- Var: %.4e' %(np.mean(abs(v[:,3:7])), np.var(abs(v[:,3:7]))))

n_final = 1900
t = np.arange(0, n_final*0.01, 0.01)
eixos = ('X', 'Y', 'Z')
plt.close('all')
for i, eixo in enumerate(eixos):
    plt.figure(eixo)
    plt.title('Evolução do Erro - Estado Inicial Aleatório - Eixo '+ eixo)
    plt.plot(t, w[0][0:n_final,i], label='Híbrido')
    plt.plot(t, w[1][0:n_final,i], label='Convencional')
    plt.grid(True)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro (m)')
    plt.legend()
    string = 'erro_' + eixo + '.png'
    plt.savefig(string)
    plt.show()