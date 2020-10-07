import numpy as np


P = 3
I = 0.0
D = 0.05

P_z = 3
I_z = 0
D_z = 0.05

P = np.array([P, P, P_z])
I = np.array([I, I, I_z])
D = np.array([D, D, D_z])

def vel_pid(state, target, cumm_error, der_error):
    position_d = target[0:5:2]
    position = state[0:5:2]
    
    cumm_error = cumm_error + (position_d - position)*0.01
    derivative = ((position_d - position) - (der_error[0]-der_error[1]))/0.01
    der_error = np.array([position_d, position])

    p = P*(position_d - position)
    i = I*cumm_error
    d = D*derivative



    err_vel = p + i + d
    # print(state[0:6], target[0:6])
    # print(err_vel[0:6])
    state_error = np.array([0, err_vel[0], 0, err_vel[1], 0, err_vel[2], 0, 0, 0, 0, 0, 0, 0, 0])

    return state_error, cumm_error, der_error

