import numpy as np

def euler_quat(ang):
    #ROTACAO 3-2-1
    phi = ang[0]
    theta = ang[1]
    psi = ang[2]
    
    cp = np.cos(phi/2)
    sp = np.sin(phi/2)
    ct = np.cos(theta/2)
    st = np.sin(theta/2)
    cps = np.cos(psi/2)
    sps = np.sin(psi/2)

    q0 = cp*ct*cps+sp*st*sps
    q1 = sp*ct*cps-cp*st*sps
    q2 = cp*st*cps+sp*ct*sps
    q3 = cp*ct*sps-sp*st*cps
    q = np.array([[q0, q1, q2, q3]]).T
    q = q/np.linalg.norm(q)
    return q


def quat_euler(q):
    phi = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2))
    theta = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
    psi = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
    phi = phi[0]
    theta = theta[0]
    psi = psi[0]
    if any(np.isnan([phi, theta, psi])):
        print('Divergencia na conversao Quaternion - Euler')
    return np.array([phi, theta, psi])


def deriv_quat(w, q):
    wx = w[0, 0]
    wy = w[1, 0]
    wz = w[2, 0]
    q0 = -1/2*(wx*q[1, 0]+wy*q[2, 0]+wz*q[3, 0])
    q1 = 1/2*(wx*q[0, 0]+wy*q[3, 0]-wz*q[2, 0])
    q2 = 1/2*(wy*q[0, 0]+wz*q[1, 0]-wx*q[3, 0])
    q3 = 1/2*(wz*q[0, 0]+wx*q[2, 0]-wy*q[1, 0])
    dq = np.array([[q0, q1, q2, q3]]).T
    return dq

def quat_rot_mat(q):
    a = q[0, 0]
    b = q[1, 0]
    c = q[2, 0]
    d = q[3, 0]
    R = np.array([[a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
                  [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
                  [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]])
    return R