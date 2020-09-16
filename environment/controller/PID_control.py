import numpy as np
import sys
sys.path.append('/home/rafael/mestrado/quadrotor_environment/')
from environment.quadrotor_env import quad, plotter
from environment.quaternion_euler_utility import euler_quat, quat_euler
from mission_control.mission_control import mission

P, I, D = 1, 0, 1
P_z, I_z, D_z = 1, 0, 1
P_a, I_a, D_a = 20, 0, 5
P_ps, I_ps, D_ps =  1, 0, 0.1

class pid_control():
    def __init__(self, drone_env):
        self.env = drone_env
        self.pid_x = pid(P, I, D) 
        self.pid_y = pid(P, I, D) 
        self.pid_z = pid(P_z, I_z, D_z) 
        
        self.pid_phi = pid(P_a, I_a, D_a) 
        self.pid_theta = pid(P_a, I_a, D_a) 
        self.pid_psi = pid(P_ps, I_ps, D_ps) 
        
        self.ang_d_ant = np.zeros(3)
        
    def lower_control(self, xd, dxd):
        
        [x, y, z] = self.env.state[0:5:2]
        [dx, dy, dz] = self.env.state[1:6:2]
        
        u_1 = self.pid_x.pid(x, dx, xd[0], dxd[0])
        u_2 = self.pid_y.pid(y, dy, xd[1], dxd[1])
        u_3 = self.pid_z.pid(z, dz, xd[2], dxd[2])

        theta_d = np.arctan(u_1/(u_3+self.env.gravity))
        
        phi_d = np.arctan(-u_2/(u_3+self.env.gravity)*np.cos(theta_d))
        
        U_1 = self.env.mass*(u_3 + self.env.gravity)/(np.cos(theta_d)*np.cos(phi_d))
        
        return U_1, phi_d, theta_d
        
        
    def upper_control(self, ang_d, v_ang_d):
        [phi, theta, psi] = self.env.ang
        [dp, dt, dps] = self.env.ang_vel
        
        u_5 = self.pid_phi.pid(phi, dp, ang_d[0], v_ang_d[0])
        u_6 = self.pid_theta.pid(theta, dt, ang_d[1], v_ang_d[1])
        u_7 = self.pid_psi.pid(psi, dps, ang_d[2], v_ang_d[2])

        sp = np.sin(phi)
        cp = np.cos(phi)

        st = np.sin(theta)
        ct = np.cos(theta)
        tt = np.tan(theta)

        b_1 = 1/self.env.J_mat[0, 0]
        b_2 = tt*sp/self.env.J_mat[1, 1]
        b_3 = tt*cp/self.env.J_mat[2, 2]
        b_4 = cp/self.env.J_mat[1, 1]
        b_5 = -sp/self.env.J_mat[2, 2]
        b_6 = sp/ct/self.env.J_mat[1, 1]
        b_7 = cp/ct/self.env.J_mat[2, 2]

        M = np.array([[b_1, b_2, b_3], 
                      [0, b_4, b_5],
                      [0, b_6, b_7]])

        [U_2, U_3, U_4] = np.dot(np.linalg.inv(M), np.array([[u_5, u_6, u_7]]).T).flatten()

        return U_2, U_3, U_4
    
    def control(self, xd = None, dxd = None, psd = 0, dpsd = 0):
        if not xd.all():
            xd = np.zeros(3)
        if not dxd.all():
            dxd = np.zeros(3)
    
        F_Z, phi_d, theta_d = self.lower_control(xd, dxd)
        ang_d = np.array([phi_d, theta_d, psi_d])
        v_ang_d = (ang_d - self.ang_d_ant)/drone.t_step
        [M_X, M_Y, M_Z] = self.upper_control(ang_d, v_ang_d)
        self.ang_d_ant = ang_d
        action = np.array([F_Z, M_X, M_Y, M_Z])
        
        return action
    
    
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
    
drone = quad(0.01, 5000, 0, direct_control=0)
mission_control = mission(drone.t_step)
mission_control.spiral_trajectory(4000, 5000, 0.3, np.pi/10, 3, np.array([0,0,0]))


initial_state = np.zeros(13)
initial_position = [0, 0, 0]
initial_velocity = [0, 0, 0]
initial_euler = np.array([0, 0, 0])/180*np.pi
initial_quaternions = euler_quat(initial_euler).flatten()
initial_angular_velocity = [0, 0, 0]


for i, (p, v) in enumerate(zip(initial_position, initial_velocity)):
    initial_state[2*i:2*i+2] = np.array([p, v])  
initial_state[6:10] = initial_quaternions
initial_state[10::] = initial_angular_velocity


drone.reset(initial_state)
controller = pid_control(drone)
plot = plotter(drone, False)
while True:
    error_mission = mission_control.get_error(drone.i*drone.t_step)
    
    xd = error_mission[0:5:2]
    dxd = error_mission[1:6:2]
    psi_d = 0
    
    action = controller.control(xd, dxd, psi_d, 0)       

    _, _, done = drone.step(action)
    plot.add()
    if done:
        plot.plot()
        break
    