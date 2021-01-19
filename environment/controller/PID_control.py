import numpy as np
import pandas as pd
import sys
import os
os.chdir('/home/rafaelcostaf/mestrado/quadrotor_environment/')
# sys.path.append()
from environment.quadrotor_env import quad, plotter
from environment.quaternion_euler_utility import euler_quat, quat_euler
from environment.controller.response_analyzer import response_analyzer
from environment.controller.target_parser import target_parse, episode_n
from mission_control.mission_control import mission
import matplotlib.pyplot as plt


test_n = 0
mission_str = 'spiral_tracking'

# P, I, D = 27.1, 0, 13
# P_z, I_z, D_z = 21, 0, 10
# P_a, I_a, D_a = 22, 0, 12
# P_ps, I_ps, D_ps =  1, 0, 0.1


P, I, D = 6, 0, 3
P_z, I_z, D_z = 6, 0, 3
P_a, I_a, D_a = 22, 0, 12
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
        
        self.log_state = []
        self.log_input = []
        self.log_target = []
        
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
    
    def control(self, xd , dxd , psd , dpsd ):
            
        self.error_mission = np.array([xd[0], dxd[0], xd[1], dxd[1], xd[2], dxd[2], psd, dpsd])
        
        F_Z, phi_d, theta_d = self.lower_control(xd, dxd)
        ang_d = np.array([phi_d, theta_d, psi_d])
        v_ang_d = (ang_d - self.ang_d_ant)/drone.t_step
        [M_X, M_Y, M_Z] = self.upper_control(ang_d, v_ang_d)
        self.ang_d_ant = ang_d
        action = np.array([F_Z, M_X, M_Y, M_Z])
        
        return action
    
    def data_logger(self):
        self.log_state.append(self.env.state)    
        self.log_target.append(self.error_mission)
        self.log_input.append(self.env.w.flatten())
        if self.env.done:
            plt.close('all')
            self.log_state = np.array(self.log_state)
            self.log_target = np.array(self.log_target)
            self.log_input = np.array(self.log_input)
            y = np.transpose(self.log_state)
            z = np.transpose(self.log_target)
            in_log = np.transpose(self.log_input)
            x = np.arange(0,len(self.log_state),1)*self.env.t_step
            
            labels = np.array([['x', r'$x_t$'],     
                               ['y', r'$y_t$'], 
                               ['z', r'$z_t$'],
                               [r'$\dot{x}$', r'$\dot{x_t}$'],
                               [r'$\dot{y}$', r'$\dot{y_t}$'],
                               [r'$\dot{z}$', r'$\dot{z_t}$']])
            
            labels_ang = np.array([r'$w_1$', r'$w_2$', r'$w_3$', r'$w_4$'])
                        
            line_style = np.array(['-', '--'])
            
            line_styles_z = np.array(['dotted', 'dashdot', 'dotted', 'dashdot', 'dotted', 'dashdot'])
            
            
            
            # POSISIONS
            plt.figure("Position")
            
            for i in range(3):
                plt.subplot(311+i)
                plt.plot(x, y[0+2*i, :], ls = '-')
                plt.plot(x, z[0+2*i, :], ls = '--')
                plt.legend(labels[i])
                if i == 2:
                    plt.xlabel('time (s)')
                if i == 1:
                    plt.ylabel('position (m)')
                plt.grid(True)
            plt.savefig('./results/pid_'+mission_str+'/position_'+str(test_n)+'.png')

            
            # VELOCITIES
            plt.figure("Velocity") 
            
            for i in range(3):
                plt.subplot(311+i)
                plt.plot(x, y[1+2*i, :], ls = '-')
                plt.plot(x, z[1+2*i, :], ls = '--')
                plt.legend(labels[3+i])
                if i == 2:
                    plt.xlabel('time (s)')
                if i == 1:
                    plt.ylabel('velocity (m/s)')
                plt.grid(True)
            plt.savefig('./results/pid_'+mission_str+'/velocities_'+str(test_n)+'.png')
            
            # ANGULAR PROP VELOCITY
            fig = plt.figure("Proppeler Angular Velocity")
            fig.text(0.5, 0.04, 'time (s)', ha='center')
            fig.text(0.04, 0.5, 'velocity (rad/s)', va='center', rotation='vertical')
            for i, data in enumerate(in_log):
                plt.subplot(411+i)
                plt.plot(x, data, label = labels_ang[i])
                plt.grid(True)                
                plt.legend()
            plt.savefig('./results/pid_'+mission_str+'/prop_angular_vel_'+str(test_n)+'.png')
            
            # 3D PLOT
            fig = plt.figure('3D plot')            
            ax = fig.gca(projection='3d')      
            
            ax.plot(y[0, :], y[2, :], y[4, :], label = 'Position')
            ax.plot(z[0, :], z[2, :], z[4, :], label = 'Target', ls='--')
            # ax.set(xlim=(-1.2,1.2), ylim=(-1.2,1.2), zlim=(0,3), )
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
            plt.legend()
            plt.savefig('./results/pid_'+mission_str+'/3D_plot_'+str(test_n)+'.png')
            plt.show()
            
            series = response_analyzer(y, target, drone.abs_sum, error_sum, drone.n)
            episode_name = 'PD '+str(target)+' '+str(ttime)+'s'
            results.insert(j, episode_name, series)
            print('Ep Number: '+str(j+1))
            self.log_state = []
            self.log_target = []
            self.log_input = []
            
            # response_analyzer(y, TARGET)        
        
        
        
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


 
drone = quad(0.01, 5000, 5, direct_control=0)
mission_control = mission(drone.t_step)
indexes = ['CE', 'EOT',
           'Over X', 'Over Y', 'Over Z',
           'Rise X', 'Rise Y', 'Rise Z',
           'Set X', 'Set Y', 'Set Z',
           'SS X', 'SS Y', 'SS Z',]

results = pd.DataFrame(0, index=indexes, columns=([]))

for j in range(episode_n()):
    m_c, ttime, target = target_parse(j)    
    if m_c==1:
        mission_control.gen_trajectory(5000, int(ttime/drone.t_step), np.array(target), )
    else:
        mission_control.spiral_trajectory(*tuple(target))
    
    
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
    
    
    drone.reset(initial_state, )
    controller = pid_control(drone)
    plot = plotter(drone, False)
    error_sum = 0
    j = 0
    while True and j < 5000:
        error_mission = mission_control.get_error(drone.i*drone.t_step)
        
        xd = error_mission[0:5:2]
        dxd = error_mission[1:6:2]
        psi_d = 0
        
        error_pos = drone.state[0:5:2]-xd 
        error_vel = drone.state[1:6:2]-dxd
        error_array = np.append(error_pos, error_vel)
        error_sum += np.sqrt(np.sum(np.square(error_pos)))    
        
        action = controller.control(xd, dxd, psi_d, 0)       
        _, _, done = drone.step(action)
        controller.data_logger()
        j += 1
        print(j)
        if done:
            break
        
results = results.T
results.to_csv('pd_results.csv')