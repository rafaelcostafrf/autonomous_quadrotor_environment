import torch
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from environment.quadrotor_env import quad, sensor
from environment.quaternion_euler_utility import deriv_quat
from environment.controller.model import ActorCritic
from environment.controller.dl_auxiliary import dl_in_gen
from mission_control.mission_control import mission

## PPO SETUP ##
time_int_step = 0.01
T = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class quad_position():
    
    def __init__(self, render, quad_model, prop_models, EPISODE_STEPS, REAL_CTRL, ERROR_AQS_EPISODES, ERROR_PATH, HOVER, M_C):
        self.REAL_CTRL = REAL_CTRL
        self.IMG_POS_DETER = False
        self.ERROR_AQS_EPISODES = ERROR_AQS_EPISODES
        self.ERROR_PATH = ERROR_PATH
        self.HOVER = HOVER
        self.M_C = M_C
        self.quad_model = quad_model
        self.prop_models = prop_models
        self.episode_n = 1
        self.time_total_sens = []
        self.T = T
        
        self.log_state = []
        self.log_target = []
        self.log_input = []
        
        self.render = render
        self.render.taskMgr.add(self.drone_position_task, 'Drone Position')
        self.render.taskMgr.add(self.drone_logger_task, 'Logging Task')
        
        # ENV SETUP
        self.env = quad(time_int_step, EPISODE_STEPS, direct_control=1, T=T)
        self.sensor = sensor(self.env)
        self.aux_dl = dl_in_gen(T, self.env.state_size, self.env.action_size)    
        self.error = []
        state_dim = self.aux_dl.deep_learning_in_size
        self.policy = ActorCritic(state_dim, action_dim=4, action_std=0).to(device)
        self.error_est_list = []
        self.error_contr_list = []
        #CONTROLLER SETUP
        try:
            self.policy.load_state_dict(torch.load('./environment/controller/PPO_continuous_solved_drone.pth',map_location=device))
            print('Saved policy loaded')
        except:
            print('Could not load policy')
            sys.exit(1)
            
        n_parameters = sum(p.numel() for p in self.policy.parameters())
        print('Neural Network Number of Parameters: %i' %n_parameters)
    
    def drone_logger_task(self, task):
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
            x = np.arange(0,len(self.log_state),1)*time_int_step
            
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
                        
            # ANGULAR PROP VELOCITY

            fig = plt.figure("Proppeler Angular Velocity")
            fig.text(0.5, 0.04, 'time (s)', ha='center')
            fig.text(0.04, 0.5, 'velocity (rad/s)', va='center', rotation='vertical')
            for i, data in enumerate(in_log):
                plt.subplot(411+i)
                plt.plot(x, data, label = labels_ang[i])
                plt.grid(True)                
                plt.legend()
            
            fig = plt.figure('3D plot')
            
            ax = fig.gca(projection='3d')        
            
            ax.plot(y[0, :], y[2, :], y[4, :], label = 'Position')
            ax.plot(z[0, :], z[2, :], z[4, :], label = 'Target', ls = '--')
            # ax.set(xlim=(-1.2,1.2), ylim=(-1.2,1.2), zlim=(0,3), )
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
            plt.legend()
            plt.show()
            
            self.log_state = []
            self.log_target = []
            self.log_input = []
        return task.cont

        
    
    
    def drone_position_task(self, task):
        if task.frame == 0 or self.env.done:
            #MISSION CONTROL SETUP
            if self.M_C:
                self.mission_control = mission(time_int_step)
                # self.mission_control.sin_trajectory(2000, 1, 0.1, np.array([0, 0, 0]), np.array([1, 1, 0]))
                self.mission_control.spiral_trajectory(2000, 0.25, 0.314, 2.5, np.array([0,0,0]))
                # self.mission_control.gen_trajectory(2, np.array([4, -5, 3]), np.array([0.1, 0.2, 0.3]))
                self.error_mission = np.zeros(14)
            else:
                self.error_mission = np.zeros(14)
            self.control_error_list = []
            self.estimation_error_list = []
            if self.HOVER:
                in_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            else:
                in_state = None
            states, action = self.env.reset(in_state)
            self.network_in = self.aux_dl.dl_input(states, action)
            self.sensor.reset()
            pos = self.env.state[0:5:2]
            ang = self.env.ang
            self.a = np.zeros(4)
            self.episode_n += 1
            print(f'Episode Number: {self.episode_n}')
        else:
            progress = self.env.i/self.env.n*100
            print(f'Progress: {progress:.2f}%', end='\r')
            action = self.policy.actor(torch.FloatTensor(self.network_in).to(device)).cpu().detach().numpy()
            self.action_log = action
            states, _, done = self.env.step(action)
            time_iter = time.time()
            _, self.velocity_accel, self.pos_accel = self.sensor.accel_int()
            self.quaternion_gyro = self.sensor.gyro_int()
            self.ang_vel = self.sensor.gyro()
            quaternion_vel = deriv_quat(self.ang_vel, self.quaternion_gyro)
            self.pos_gps, self.vel_gps = self.sensor.gps()
            self.quaternion_triad, _ = self.sensor.triad()
            self.time_total_sens.append(time.time() - time_iter)
            if self.M_C:
                self.error_mission = self.mission_control.get_error(self.env.i*self.env.t_step)
            else:
                self.error_mission = np.zeros(14)
            #SENSOR CONTROL
            pos_vel = np.array([self.pos_accel[0], self.velocity_accel[0],
                                self.pos_accel[1], self.velocity_accel[1],
                                self.pos_accel[2], self.velocity_accel[2]])

            if self.REAL_CTRL:
                self.network_in = self.aux_dl.dl_input(states-self.error_mission, [action])
            else:
                states_sens = [np.concatenate((pos_vel, self.quaternion_gyro, quaternion_vel))]-self.error_mission                  
                self.network_in = self.aux_dl.dl_input(states_sens, [action])
            
            pos = self.env.state[0:5:2]
            ang = self.env.ang
            for i, w_i in enumerate(self.env.w):
                self.a[i] += (w_i*time_int_step )*180/np.pi/10
    
        ang_deg = (ang[2]*180/np.pi, ang[0]*180/np.pi, ang[1]*180/np.pi)
        pos = (0+pos[0], 0+pos[1], 5+pos[2])
        
        # self.quad_model.setHpr((45, 0, 45))
        # self.quad_model.setPos((5, 5, 6))
        self.quad_model.setPos(*pos)
        self.quad_model.setHpr(*ang_deg)
        for prop, a in zip(self.prop_models, self.a):
            prop.setHpr(a, 0, 0)
        return task.cont