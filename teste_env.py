from environment.quadrotor_env import quad, plotter
import numpy as np

env = quad(0.01, 10000, euler=0, direct_control=1, T=1)

int_st = np.zeros(13)
int_st[6] = 1
int_st[1] = 1
int_st[2] = 1
int_st[4] = 1
_, _ = env.reset(int_st)
a_0 = np.zeros(4)
while True:
    st, rw, dn = env.step(a_0)
    print(st, rw, dn, env.solved)
