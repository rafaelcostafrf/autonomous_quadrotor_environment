import numpy as np
import sys
sys.path.append('/home/rafael/mestrado/quadrotor_environment/')
from environment.quadrotor_env import quad, plotter

class pid_control():
    def __init__(self):
        self.m_t = 1.03
        self.g = 9.82
        
        P, I, D = 1, 0, 0
        P_t, I_t, D_t = 1, 0, 0
        
        self.pid_phi = pid(P, I, D)
        self.pid_theta = pid(P, I, D)
        self.pid_psi = pid(P, I, D)
        
        self.pid_x = pid(P_t, I_t, D_t)
        self.pid_y = pid(P_t, I_t, D_t)
        self.pid_z = pid(P_t, I_t, D_t)
        
    def lower_control(self, x, dx, y, dy, z, dz, x_d, dx_d, y_d, dy_d, z_d, dz_d):
        a_x = self.pid_x.pid(x, x_d, dx, dx_d)
        a_y = self.pid_y.pid(y, y_d, dy, dy_d)
        a_z = self.pid_z.pid(z, z_d, dz, dz_d)        
        theta_d = np.arctan2(a_x, a_z+self.g)
        phi_d = np.arctan2(-a_y*np.cos(theta_d), a_z+self.g)
        U_1 = self.m_t*(a_z+self.g)/np.cos(theta_d)/np.cos(phi_d)
        return theta_d, phi_d, U_1
        
    def upper_control(self, phi, dphi, theta, dtheta, psi, dpsi, phi_d, dphi_d, theta_d, dtheta_d, psi_d, dpsi_d):
        c_phi = self.pid_phi.pid(phi, phi_d, dphi, dphi_d)
        c_theta = self.pid_theta.pid(theta, theta_d, dtheta, dtheta_d)
        c_psi = self.pid_psi.pid(psi, psi_d, dpsi, dpsi_d)
        return c_phi, c_theta, c_psi
        
class pid():
    def __init__(self, P, I, D, timestep=0.01):
        self.ix = 0
        self.p = P
        self.i = I
        self.d = D
        self.ts = timestep
        
    def pid(self, x, dx, x_d, dx_d=0):
        self.ix = self.ix+(x_d-x)*self.ts
        control = self.p*(x_d-x)+self.d*(dx_d-dx)-self.i*(self.ix)
        return control
    
drone = quad(0.01, 10000, 0, direct_control=0)
drone.reset()
controller = pid_control()
plot = plotter(drone, False)
while True:
    x, dx, y, dy, z, dz  = drone.state[0:6]
    phi, theta, psi = drone.ang
    dphi, dtheta, dpsi = drone.state[-3:]
    theta_d, phi_d, U1 = controller.lower_control(x, dx, y, dy, z, dz, 0, 0, 0, 0, 0, 0)
    U2, U3, U4 = controller.upper_control(phi, dphi, theta, dtheta, psi, dpsi, phi_d, 0, theta_d, 0, 0, 0)
    action = np.array([U1, U2, U3, U4])
    _, _, done = drone.step(action)
    plot.add()
    if done:
        plot.plot()
        break
    