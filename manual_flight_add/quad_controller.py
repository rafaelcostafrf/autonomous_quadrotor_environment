import sys
import torch
import numpy as np 

from environment.quadrotor_env_opt import quad, sensor
from environment.quaternion_euler_utility import deriv_quat, euler_quat, quat_euler_2
from environment.controller.model import ActorCritic
from environment.controller.dl_auxiliary import dl_in_gen
from visual_landing.rl_reward_fuction import visual_reward

T = 5
TIME_STEP = 0.01
TOTAL_STEPS = 4000
VELOCITY_SCALE = [3, 3, 3]

#CONTROL POLICY
AUX_DL = dl_in_gen(T, 13, 4)
state_dim = AUX_DL.deep_learning_in_size
CRTL_POLICY = ActorCritic(state_dim, action_dim=4, action_std=0)
try:
    CRTL_POLICY.load_state_dict(torch.load('./environment/controller/PPO_continuous_solved_drone.pth'))
    print('Saved Control policy loaded')
except:
    print('Could not load Control policy')
    sys.exit(1)  
    
class quad_sim():
    def __init__(self, render ):
        self.done = False
        self.render = render
        
        self.quad_model = render.quad_model
        self.prop_models = render.prop_models
        self.a = np.zeros(4)
        
        #CONTROLLER POLICY
        self.quad_env = quad(TIME_STEP, TOTAL_STEPS, 1, T)
        self.sensor = sensor(self.quad_env)  
        states, action = self.quad_reset_random()
        self.sensor.reset()
        self.aux_dl =  dl_in_gen(T, 13, 4)
        self.control_network_in = self.aux_dl.dl_input(states, action)
        self.last_shaping = 0
        
        #TASK MANAGING
        self.visual_done = False

        #MARKER POSITION
        self.internal_frame = 0
        self.crtl_action = np.zeros(4)
        
        #CAMERA SETUP
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        self.d_angs = np.array([0, 0, 0])
        
    def quad_reset_random(self):
        random_marker_position = np.random.normal([0, 0], 0.8)
        self.render.checker.setPos(*tuple(random_marker_position), 0.001)
        self.marker_position = np.append(random_marker_position, 0.001)
        
        
        quad_random_z = -5*np.random.random()+1
        quad_random_xy = self.marker_position[0:2]+(np.random.random(2)-0.5)*quad_random_z/7*5/1.2
        initial_state = np.array([quad_random_xy[0], 0, quad_random_xy[1], 0, quad_random_z, 0, 1, 0, 0, 0, 0, 0, 0])
        states, action = self.quad_env.reset(initial_state)
        self.sensor.reset()
        self.d_angs = np.array([0, 0, 0])
        return states, action
        
    def sensor_sp(self):
            _, self.velocity_accel, self.pos_accel = self.sensor.accel_int()
            self.quaternion_gyro = self.sensor.gyro_int()
            self.ang_vel = self.sensor.gyro()
            quaternion_vel = deriv_quat(self.ang_vel, self.quaternion_gyro)
            self.pos_gps, self.vel_gps = self.sensor.gps()
            self.quaternion_triad, _ = self.sensor.triad()
            pos_vel = np.array([self.pos_accel[0], self.velocity_accel[0],
                                self.pos_accel[1], self.velocity_accel[1],
                                self.pos_accel[2], self.velocity_accel[2]])
            states_sens = np.array([np.concatenate((pos_vel, self.quaternion_gyro, quaternion_vel))   ])
            return states_sens
  
        
    def render_position(self, coordinates, marker_position):

        self.render.checker.setPos(*tuple(marker_position))

        pos = coordinates[0:3]
        ang = coordinates[3:6]
        w = coordinates[6::]

        for i, w_i in enumerate(w):
            self.a[i] += (w_i*TIME_STEP)*180/np.pi/10
        ang_deg = (ang[2]*180/np.pi, ang[0]*180/np.pi, ang[1]*180/np.pi)
        pos = (0+pos[0], 0+pos[1], 5+pos[2])
        
        self.quad_model.setPos(*pos)
        self.quad_model.setHpr(*ang_deg)
        self.render.dlightNP.setPos(*pos)
        for prop, a in zip(self.prop_models, self.a):
            prop.setHpr(a, 0, 0)
           
    def step(self, x_vel, y_vel, z_vel):
        states_sens = self.sensor_sp()
        
        # CONTROL DIFFERENCE
        error = np.array([[0, x_vel, 0, y_vel, 0, z_vel, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.control_network_in = self.aux_dl.dl_input(states_sens-error, [self.crtl_action])
        crtl_network_in = torch.FloatTensor(self.control_network_in).to('cpu')

        self.crtl_action = CRTL_POLICY.actor(crtl_network_in).cpu().detach().numpy()

        states, _, _ = self.quad_env.step(self.crtl_action)

        coordinates = np.concatenate((states[0, 0:5:2], self.quad_env.ang, np.zeros(4))) 

        self.render_position(coordinates, self.marker_position)
        
                
        _, self.last_shaping, self.visual_done = visual_reward(TOTAL_STEPS, self.marker_position, self.quad_env.state[0:5:2], self.quad_env.state[1:6:2], np.zeros(3), self.last_shaping, self.internal_frame, self.quad_env.ang)

        if self.visual_done:
            self.quad_reset_random()        
