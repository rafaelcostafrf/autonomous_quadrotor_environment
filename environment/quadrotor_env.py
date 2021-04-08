from scipy import integrate
import numpy as np
from environment.quaternion_euler_utility import euler_quat, quat_euler, deriv_quat, quat_rot_mat
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

"""""
QUADROTOR ENVIRONMENT
DEVELOPED BY: 
    RAFAEL COSTA FERNANDES
    PROGRAMA DE PÓS GRADUAÇÃO EM ENGENHARIA MECÂNICA, UNIVERSIDADE FEDERAL DO ABC
    SP - SANTO ANDRÉ - BRASIL

FURTHER DOCUMENTATION ON README.MD
"""""

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.preamble':[
        '\DeclareUnicodeCharacter{2212}{-}']
})

## SIMULATION BOUNDING BOXES ##
BB_POS = 5
BB_VEL = 10
BB_CONTROL = 9
BB_ANG = np.pi/2


# QUADROTOR MASS AND GRAVITY VALUE
M, G = 1.03, 9.82

# AIR DENSITY
RHO = 1.2041

#DRAG COEFFICIENT
C_D = 1.1

# ELETRIC MOTOR THRUST AND MOMENT
K_F = 1.435e-5
K_M = 2.4086e-7
I_R = 5e-5
T2WR = 2


# INERTIA MATRIX
J = np.array([[16.83e-3, 0, 0],
              [0, 16.83e-3, 0],
              [0, 0, 28.34e-3]])

# ELETRIC MOTOR DISTANCE TO CG
D = 0.26

#PROJECTED AREA IN X_b, Y_b, Z_b
BEAM_THICKNESS = 0.05
A_X = BEAM_THICKNESS*2*D
A_Y = BEAM_THICKNESS*2*D
A_Z = BEAM_THICKNESS*2*D*2
A = np.array([[A_X,A_Y,A_Z]]).T


## REWARD PARAMETERS ##
SOLVED_REWARD = 20
BROKEN_REWARD = -20
SHAPING_WEIGHT = 5
SHAPING_INTERNAL_WEIGHTS = [15, 4, 1]

# CONTROL REWARD PENALITIES #
P_C = 0.003
P_C_D = 0

## TARGET STEADY STATE ERROR ##
TR = [0.005, 0.01, 0.1]
TR_P = [3, 2, 1]

## ROBUST CONTROL PARAMETERS
class robust_control():
    def __init__(self):
        self.D_KF = 0.1
        self.D_KM = 0.1
        self.D_M = 0.3
        self.D_IR = 0.1
        self.D_J = np.ones(3) * 0.1
        self.reset()
        self.gust_std = [[5], [5], [2]]
        self.gust_period = 500 # integration steps
        self.i_gust = 0
        self.gust = np.zeros([3, 1])

    def reset(self):
        self.episode_kf = np.random.random(4) * self.D_KF
        self.episode_m = np.random.normal(0, self.D_M, 1)
        self.episode_ir = np.random.random(4) * self.D_IR
        self.episode_J = np.eye(3)*np.random.normal(np.zeros(3), self.D_J, [3])

    def wind(self, i):
        index = (i % self.gust_period) - 1
        if index % self.gust_period == 0:
            self.last_gust = self.gust
            self.gust = np.random.normal(np.zeros([3, 1]), self.gust_std, [3, 1])
            self.linear_wind_change = np.linspace(self.last_gust, self.gust, self.gust_period)
        return self.linear_wind_change[index]

class quad():
    def __init__(self, t_step, n, training = True, euler=0, direct_control=1, T=1, clipped = True):        
        """"
        inputs:
            t_step: integration time step 
            n: max timesteps
            euler: flag to set the states return in euler angles, if off returns quaternions
            deep learning:
                deep learning flag: If on, changes the way the env. outputs data, optimizing it to deep learning use.
                T: Number of past history of states/actions used as inputs in the neural network
                debug: If on, prints a readable reward funcion, step by step, for a simple reward weight debugging.
        
        """
        self.clipped = clipped


        if training:
            self.ppo_training = True
        else:
            self.ppo_training = False
        
        
        self.mass = M
        self.gravity = G
        
        self.i = 0
        self.T = T                                              #Initial Steps
        
        self.bb_cond = np.array([BB_VEL,
                                 BB_VEL,
                                 BB_VEL,
                                 BB_ANG, BB_ANG, 3/4*np.pi,
                                 BB_VEL*2, BB_VEL*2, BB_VEL*2])       #Bounding Box Conditions Array
        if not self.ppo_training:
            self.bb_cond = self.bb_cond*1
            
        #Quadrotor states dimension
        self.state_size = 13       
        
        #Quadrotor action dimension                            
        self.action_size = 4                                    
        
        #Env done Flag
        self.done = True                                        
        
        #Env Maximum Steps
        self.n = n+self.T

                                               
        self.t_step = t_step

        
        #Neutral Action (used in reset and absolute action penalty) 
        if direct_control:
            self.zero_control = np.ones(4)*(2/T2WR - 1)             
        else:
            self.zero_control = np.array([M*G, 0, 0, 0])
            
        self.direct_control_flag = direct_control
        
        self.ang_vel = np.zeros(3)
        self.prev_ang = np.zeros(3)
        self.J_mat = J
        
        #Absolute sum of control efforts over the episode
        self.abs_sum = 0
        
        self.d_xx = np.linspace(0, D, 10)
        self.d_yy = np.linspace(0, D, 10)
        self.d_zz = np.linspace(0, D, 10)
        
        self.robust_parameters = robust_control()
        self.robust_control = False

        ev_cd = 'Training' if self.ppo_training else 'Eval'
        ct_cd = ' with robust environment' if self.robust_control else ''
        print('Environment Condition: ' + ev_cd + ct_cd)
        
    def seed(self, seed):
        """"
        Set random seeds for reproducibility
        """       
        np.random.seed(seed)       
    
    
    
    def f2w(self,f,m):
        """""
        Translates F (Thrust) and M (Body x, y and z moments) into eletric motor angular velocity (rad/s)
        input:
            f - thrust 
            m - body momentum in np.array([[mx, my, mz]]).T
        outputs:
            F - Proppeler Thrust - engine 1 to 4
            w - Proppeler angular velocity - engine 1 to 4
            F_new - clipped thrust (if control surpasses engine maximum)
            M_new - clipped momentum (same as above)
        """""
        x = np.array([[K_F, K_F, K_F, K_F],
                      [-D*K_F, 0, D*K_F, 0],
                      [0, D*K_F, 0, -D*K_F],
                      [-K_M, +K_M, -K_M, +K_M]])

        y = np.array([f, m[0,0], m[1,0], m[2,0]])
        
        u = np.linalg.solve(x, y)

        if self.clipped:
            u = np.clip(u, 0, T2WR*M*G/4/K_F)
            w_1 = np.sqrt(u[0])
            w_2 = np.sqrt(u[1])
            w_3 = np.sqrt(u[2])
            w_4 = np.sqrt(u[3])   
        else:
            modules = np.zeros(4)
            for k in range(4):
                modules[k] = -1 if u[k] < 0 else 1
            w_1 = np.sqrt(np.abs(u[0]))*modules[0]
            w_2 = np.sqrt(np.abs(u[1]))*modules[1]
            w_3 = np.sqrt(np.abs(u[2]))*modules[2]
            w_4 = np.sqrt(np.abs(u[3]))*modules[3]
            
        w = np.array([[w_1,w_2,w_3,w_4]]).T

        if self.robust_control:
            u -= u*self.robust_parameters.episode_kf

        FM_new = np.dot(x, u)
        
        F_new = FM_new[0]
        M_new = FM_new[1:4]
        
        step_effort = (u*K_F/(T2WR*M*G/4)*2)-1
        
        return step_effort, w, F_new, M_new
        
    def f2F(self, f_action):
        """""
        Translates Proppeler thrust to body trhust and body angular momentum.
        input:
            f_action - proppeler thrust written as np.array([f1, f2, f3, f4])
                        the proppeler thrust if normalized in [-1, 1] domain, where -1 is 0 thrust and 1 is maximum thrust 
        output:
            w - proppeler angular velocity
            F_new - body thrust
            M_new - body angular momentum
        """""
        f = (f_action + 1) * T2WR * M * G / 8

        w = np.array([[np.sqrt(f[0]/K_F)],
                      [np.sqrt(f[1]/K_F)],
                      [np.sqrt(f[2]/K_F)],
                      [np.sqrt(f[3]/K_F)]])

        if self.robust_control:
            f = f - self.robust_parameters.episode_kf * f
        
        F_new = np.sum(f)
        M_new = np.array([[(f[2]-f[0])*D],
                          [(f[1]-f[3])*D],
                          [(-f[0]+f[1]-f[2]+f[3])*K_M/K_F]])
        return w, F_new, M_new
    
    def drone_eq(self, t, x, action):
        
        """"
        Main differential equation, not used directly by the user, rather used in the step function integrator.
        Dynamics based in: 
            MODELAGEM DINÂMICA E CONTROLE DE UM VEÍCULO AÉREO NÃO TRIPULADO DO TIPO QUADRIRROTOR 
            by ALLAN CARLOS FERREIRA DE OLIVEIRA
            BRASIL, SP-SANTO ANDRÉ, UFABC - 2019
        Incorporates:
            Drag Forces, Gyroscopic Forces
            In indirect mode: Force clipping (preventing motor shutoff and saturates over Thrust to Weight Ratio)
            In direct mode: maps [-1,1] to forces [0,T2WR*G*M/4]
        """
        if self.direct_control_flag:
            self.w, f_in, m_action = self.f2F(action)
        else:            
            f_in = action[0]
            m_action = np.array([action[1::]]).T            

        
        #BODY INERTIAL VELOCITY                
        vel_x = x[1]
        vel_y = x[3]
        vel_z = x[5]
        
        #QUATERNIONS
        q0 = x[6]
        q1 = x[7]
        q2 = x[8]
        q3 = x[9]
        
        #BODY ANGULAR VELOCITY
        w_xx = x[10]
        w_yy = x[11]
        w_zz = x[12]      

        #QUATERNION NORMALIZATION (JUST IN CASE)
        q = np.array([[q0, q1, q2, q3]]).T
        q = q/np.linalg.norm(q)
        
        # DRAG FORCES ESTIMATION (BASED ON BODY VELOCITIES)
        self.mat_rot = quat_rot_mat(q)

        v_inertial = np.array([[vel_x, vel_y, vel_z]]).T
        if self.robust_control:
            wind = self.robust_parameters.wind(self.i)
            v_inertial += wind

        v_body = np.dot(self.mat_rot.T, v_inertial)
        f_drag = -0.5*RHO*C_D*np.multiply(A,np.multiply(abs(v_body),v_body))
        
        # DRAG MOMENTS ESTIMATION (BASED ON BODY ANGULAR VELOCITIES)
        
        #Discretization over 10 steps (linear velocity varies over the body)
        m_x = 0
        m_y = 0
        m_z = 0
        for xx,yy,zz in zip(self.d_xx, self.d_yy, self.d_zz):
            m_x += -RHO*C_D*BEAM_THICKNESS*D/10*(abs(xx*w_xx)*(xx*w_xx))*xx
            m_y += -RHO*C_D*BEAM_THICKNESS*D/10*(abs(yy*w_yy)*(yy*w_yy))*yy
            m_z += -2*RHO*C_D*BEAM_THICKNESS*D/10*(abs(zz*w_zz)*(zz*w_zz))*zz
        
        m_drag = np.array([[m_x],
                           [m_y],
                           [m_z]])
        
        #GYROSCOPIC EFFECT ESTIMATION (BASED ON ELETRIC MOTOR ANGULAR VELOCITY)
        if self.robust_control:
            ir = I_R*(np.ones(4)+self.robust_parameters.episode_ir)
            omega_r = (-self.w[0]*ir[0]+self.w[1]*ir[1]-self.w[2]*ir[2]+self.w[3]*ir[3])[0]
        else:
            omega_r = (-self.w[0]+self.w[1]-self.w[2]+self.w[3])[0]*I_R

        m_gyro = np.array([[-w_xx*omega_r],
                           [+w_yy*omega_r],
                           [0]])

        #BODY FORCES
        self.f_in = np.array([[0, 0, f_in]]).T
        self.f_body = self.f_in+f_drag


        #BODY FORCES ROTATION TO INERTIAL
        self.f_inertial = np.dot(self.mat_rot, self.f_body)
        
        #INERTIAL ACCELERATIONS
        if self.robust_control:
            quad_m = M * (1 + self.robust_parameters.episode_m)
        else:
            quad_m = M

        accel_x = self.f_inertial[0, 0]/quad_m
        accel_y = self.f_inertial[1, 0]/quad_m
        accel_z = self.f_inertial[2, 0]/quad_m-G
        self.accel = np.array([[accel_x, accel_y, accel_z]]).T

        # self.accelerometer_read = self.f_body/quad_m
        self.accelerometer_read = self.mat_rot.T @ (self.accel.flatten() + np.array([0, 0, -G]))

        #BODY MOMENTUM
        W = np.array([[w_xx],
                 [w_yy],
                 [w_zz]])

        m_in = m_action + m_gyro + m_drag - np.cross(W.flatten(), np.dot(J, W).flatten()).reshape((3,1))

        #INERTIAL ANGULAR ACCELERATION
        if self.robust_control:
            self.inv_j = np.linalg.inv(J + J*self.robust_parameters.episode_J)
        else:
            self.inv_j = np.linalg.inv(J)
        accel_ang = np.dot(self.inv_j, m_in).flatten()
        accel_w_xx = accel_ang[0]
        accel_w_yy = accel_ang[1]
        accel_w_zz = accel_ang[2]
        
        #QUATERNION ANGULAR VELOCITY (INERTIAL)
   
        self.V_q = deriv_quat(W, q).flatten()
        dq0=self.V_q[0]
        dq1=self.V_q[1]
        dq2=self.V_q[2]
        dq3=self.V_q[3]

        
        # RESULTS ORDER:
        # 0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 q0, 7 q1, 8 q2, 9 q3, 10 w_xx, 11 w_yy, 12 w_zz
        out = np.array([vel_x, accel_x,
                         vel_y, accel_y,
                         vel_z, accel_z,
                         dq0, dq1, dq2, dq3,
                         accel_w_xx, accel_w_yy, accel_w_zz])
        return out

    def reset(self, det_state = None):
        
        """""
        inputs:_, self.w, f_in, m_action = self.f2w(f_in, m_action)
            det_state: 
                if == 0 randomized initial state
                else det_state is the actual initial state, depending on the euler flag
                if euler flag is on:
                    [x, dx, y, dy, z, dz, phi, theta, psi, w_xx, w_yy, w_zz]
                if euler flag is off:
                    [x, dx, y, dy, z, dz, q_0, q_1, q_2, q_3, w_xx, w_yy, w_zz]
        outputs:
            previous_state: system's initial state
        """""
        state = []
        action = []
        self.action_hist = []

        self.robust_parameters.reset()

        self.solved = 0
        self.done = False
        self.i = 0   
        self.prev_shaping = None
        self.previous_state = np.zeros(self.state_size)
        self.abs_sum = 0
        
        if det_state is not None:        
            self.previous_state = det_state
            q = np.array([self.previous_state[6:10]]).T
            self.ang = quat_euler(q)
        else:
            self.ang = np.random.rand(3)-0.5
            Q_in = euler_quat(self.ang)
            self.previous_state[0:5:2] = np.clip((np.random.normal([0, 0, 0], 2)), -BB_POS/2, BB_POS/2)
            self.previous_state[1:6:2] = np.clip((np.random.normal([0, 0, 0], 2)), -BB_VEL/2, BB_VEL/2)
            self.previous_state[6:10] = Q_in.T
            self.previous_state[10:13] = np.clip((np.random.normal([0, 0, 0], 2)), -BB_VEL*1.5, BB_POS*1.5)
        
        for i in range(self.T):
            self.action = self.zero_control
            self.action_hist.append(self.action)

            state_t, reward, done = self.step(self.action)
            state.append(state_t.flatten())
            action.append(self.zero_control)
        return np.array(state), np.array(action)
    


    def step(self, action):
        
        """""
        inputs:
            action: action to be applied on the system
        outputs:
            state: system's state in t+t_step actuated by the action
            done: False, else the system has breached any bounding box, exceeded maximum timesteps, or reached goal.
        """""
        self.i += 1
                
        
        if self.direct_control_flag:
            self.action = np.clip(action,-1,1)
            u = self.action
            self.clipped_action = self.action
            self.step_effort = self.action
        else:
            self.action = action
            self.step_effort, self.w, f_in, m_action = self.f2w(action[0], np.array([action[1::]]).T)
            self.clipped_action = np.append([f_in], m_action)
            u = self.clipped_action
       
        self.action_hist.append(self.clipped_action)
        
        self.y = (integrate.solve_ivp(self.drone_eq, (0, self.t_step), self.previous_state, args=(u, ))).y
        
        self.state = np.transpose(self.y[:, -1])
        self.quat_state = np.array([np.concatenate((self.state[0:10], self.V_q))])
        
        q = np.array([self.state[6:10]]).T
        q = q/np.linalg.norm(q)

        self.ang = quat_euler(q)
        self.ang_vel = (self.ang - self.prev_ang)/self.t_step
        self.prev_ang = self.ang
        self.previous_state = self.state
        self.done_condition()
        self.reward_function()
        self.control_effort()
        return self.quat_state, self.reward, self.done

    def done_condition(self):
        
        """""
        Checks if bounding boxes done condition have been met
        """""
        
        cond_x = np.concatenate((self.state[1:6:2], self.ang, self.state[-3:]))
        for x, c in zip(np.abs(cond_x), self.bb_cond):
            if  x >= c:
                self.done = True
            
    def reward_function(self, debug=0):
        
        """""
        Reward Function: Working with PPO great results.
        Shaping with some ideas based on Continuous Lunar Lander v.2 gym environment:
            https://gym.openai.com/envs/LunarLanderContinuous-v2/
        
        """""
        
        self.reward = 0
        
        velocity = self.state[1:6:2]
        euler_angles = self.ang
        psi = self.ang[2]
        body_ang_vel = self.state[-3:]
        action = self.action

        
        shaping = -SHAPING_WEIGHT/np.sum(SHAPING_INTERNAL_WEIGHTS)*(SHAPING_INTERNAL_WEIGHTS[0]*norm(velocity/BB_VEL)+
                        SHAPING_INTERNAL_WEIGHTS[1]*norm(psi/4)+
                        SHAPING_INTERNAL_WEIGHTS[2]*norm(euler_angles[0:2]/BB_ANG))
        
        
        #CASCADING REWARDS
        r_state = np.concatenate((velocity, [psi]))  

        for TR_i, TR_Pi in zip(TR, TR_P): 
            if norm(r_state) < norm(np.ones(len(r_state))*TR_i):
                shaping += TR_Pi
                if norm(euler_angles[0:2]) < norm(np.ones(2)*TR_i*4):
                    shaping += TR_Pi
                break
        
        
        if self.prev_shaping is not None:
            self.reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        
        #ABSOLUTE CONTROL PENALTY
        

        ## TOTAL REWARD SHAPING ##
        abs_control = -np.sum(np.square(action - self.zero_control)) * P_C
        self.reward += + abs_control 
        
        #SOLUTION ACHIEVED?
        self.target_state = 9*(TR[0]**2)
        self.current_state = np.sum(np.square(np.concatenate((velocity, euler_angles, body_ang_vel))))      
        
        
        
        if self.current_state < self.target_state:
            self.reward += SOLVED_REWARD
            self.solved = 1
            if self.ppo_training:
                self.done = True              
        elif self.i >= self.n:
            self.reward = self.reward
            self.solved = 0   
            self.done=True
        elif self.done:
            self.reward += BROKEN_REWARD
            self.solved = 0            
         
    def control_effort(self):
        instant_effort = np.sqrt(np.sum(np.square(self.step_effort-np.array([0*M*G, 0, 0, 0]))))
        self.abs_sum += instant_effort
        
class sensor():
    
    """Sensor class - simulates onboard sensors, given standard deviation and bias.
    Aimed to simulate kallman filters or to execute robust control, etc.
    Self explanatory, adds standard deviation noise and bias to quadrotor real state.
    
    """
    
    def __init__(self, env,
                 accel_std = 0.1, accel_bias_drift = 0.0005,
                 gyro_std = 0.035, gyro_bias_drift = 0.00015,
                 magnet_std = 15, magnet_bias_drift = 0.075,
                 gps_std_p = 1.71, gps_std_v=0.5):
        
        self.std = [accel_std, gyro_std, magnet_std, gps_std_p, gps_std_v]
        self.b_d = [accel_bias_drift, gyro_bias_drift, magnet_bias_drift]
        self.quad = env
        self.error = True
        self.bias_reset()
        self.R = np.eye(3)

    def bias_reset(self):        
        self.a_std = self.std[0]*self.error
        self.a_b_d = (np.random.random()-0.5)*2*self.b_d[0]*self.error        
        self.g_std = self.std[1]*self.error
        self.g_b_d = (np.random.random()-0.5)*2*self.b_d[1]*self.error        
        self.m_std = self.std[2]*self.error
        self.m_b_d = (np.random.random()-0.5)*2*self.b_d[2]*self.error        
        self.gps_std_p = self.std[3]*self.error
        self.gps_std_v = self.std[4]*self.error
    
        
    def accel(self):
    
        self.a_b_accel = self.a_b_accel + self.a_b_d*self.quad.t_step

        read_error = np.random.normal(self.a_b_accel, self.a_std, 3)

        read_accel_body = self.quad.accelerometer_read.flatten()

        return read_accel_body+read_error
    
    
    def gyro(self):
        
        self.g_b = self.g_b + self.g_b_d*self.quad.t_step
        
        read_error = np.random.normal(self.g_b, self.g_std, 3)
        read_gyro = self.quad.state[-3:].flatten()
        return read_error+read_gyro        
            
    def reset(self):
        self.a_b_grav = 0
        self.a_b_accel = 0
        self.m_b = 0
        self.g_b = 0
        self.acceleration_t0 = np.zeros(3)
        self.position_t0 = self.quad.state[0:5:2]
        self.velocity_t0 = self.quad.state[1:6:2]
        self.quaternion_t0 = self.quad.state[6:10]
        self.bias_reset()

    
    def gps(self):
        read_error_pos = np.random.normal(0, self.gps_std_p, 3)
        read_error_vel = np.random.normal(0, self.gps_std_v, 3)
        gps_pos = self.quad.state[0:5:2].flatten()
        gps_vel = self.quad.state[1:6:2].flatten()
        return read_error_pos+gps_pos, read_error_vel+gps_vel   
    
    def triad(self):
        gravity_vec = np.array([0, 0, -G])
        magnet_vec = np.array([-4047, 12911, -9899])*0.01
        #Magnetic Vector of Santo André - Brasil in MiliGauss
        #https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm

        
        #Gravity vector as read from body sensor
        induced_acceleration = self.quad.f_in.flatten() - (self.R @ np.array([[0, 0, -G]]).T).flatten()
        gravity_body = self.accel() - induced_acceleration

        #Magnetic Field vector as read from body sensor
        magnet_body = self.quad.mat_rot.T @ (np.random.normal(magnet_vec, self.m_std))
      
        #Accel vector is more accurate
        #Body Coordinates
        gravity_body = gravity_body / np.linalg.norm(gravity_body)

        magnet_body = magnet_body / np.linalg.norm(magnet_body)


        t1b = gravity_body/np.linalg.norm(gravity_body)

        t2b = np.cross(gravity_body, magnet_body)
        t2b = t2b/np.linalg.norm(t2b)
        
        t3b = np.cross(t1b, t2b)
        t3b = t3b/np.linalg.norm(t3b)
        
        tb = np.vstack((t1b, t2b, t3b)).T

        #Inertial Coordinates
        gravity_vec = gravity_vec/np.linalg.norm(gravity_vec)
        magnet_vec = magnet_vec / np.linalg.norm(magnet_vec)

        t1i = gravity_vec/np.linalg.norm(gravity_vec)

        t2i = np.cross(gravity_vec, magnet_vec)
        t2i = t2i/np.linalg.norm(t2i)
        
        t3i = np.cross(t1i, t2i)
        t3i = t3i/np.linalg.norm(t3i)
        
        ti = np.vstack((t1i, t2i, t3i)).T
        self.R = tb @ ti.T

        q = Rotation.from_matrix(self.R.T).as_quat()
        q = np.concatenate(([q[3]], q[0:3]))
        return q, self.R
        
        
    def accel_int(self):

        accel_body = self.accel()
        _, R = self.triad()

        acceleration = R.T @ accel_body + np.array([0, 0, G])

        velocity = self.velocity_t0 + acceleration*self.quad.t_step
        position = self.position_t0 + velocity*self.quad.t_step
        
        self.acceleration_t0 = acceleration
        self.velocity_t0 = velocity
        self.position_t0 = position


        return acceleration, velocity, position
    
    def gyro_int(self):
        w = self.gyro()
        q = self.quaternion_t0
        V_q = deriv_quat(w, q).flatten()       
        for i in range(len(q)):
            q[i] = q[i] + V_q[i]*self.quad.t_step
        self.quaternion_t0 = q/np.linalg.norm(q)
        return q
        
        
class plotter(): 
        
    """""
    Render Class: Saves state and time until plot function is called.
                    Optionally: Plots a 3D graph of the position, with optional target position.
    
    init input:
        env: 
            class - quadrotor enviorment
        depth_plot:
            boolean - plot coordinates over time on 3D space
            
            
    add: saves a state and a time

    clear: clear memory

    plot: plot saved states     
    """""   
    
    def __init__(self, env, velocity_plot = False, depth_plot=False):        
        plt.close('all')
        self.figure = plt.figure('States')
        self.depth_plot = depth_plot
        self.env = env
        self.states = []
        self.times = []
        self.print_list = range(13)
        if velocity_plot:
            self.plot_labels = ['$\dot X$', '$\dot Y$', '$\dot Z$',
                                '', '', '',
                                '$\phi$', '$\\theta$', '$\psi$', 
                                '$T_{MH,1}$', '$T_{MH,2}$', '$T_{MH,3}$', '$T_{MH,4}$']
            self.axis_labels = ['Velocidade (ms)', 'Velocidade (ms)', 'Velocidade (ms)',
                                'Velocidade (ms)', 'Velocidade (ms)', 'Velocidade (ms)',
                                'Atitude (rad)', 'Atitude (rad)', 'Atitude (rad)', 
                                'Empuxo (N)', 'Empuxo (N)', 'Empuxo (N)', 'Empuxo (N)']
            self.depth_plot = False
        else:
            self.plot_labels = ['x', 'y', 'z',
                                '$\phi$', '$\theta$', '$\psi$', 
                                '$u_1$', '$u_2$', '$u_3$', '$u_4$']
            
        self.line_styles = ['-', '-', '-',
                            '--', '--', '--',
                            '--', '--', '--', 
                            ':', ':', ':', ':']
        self.color = ['r', 'g', 'b',
                      'r', 'g', 'b',
                      'r', 'g', 'b',
                      'r', 'g', 'b', 'c']
        self.plot_place = [0, 0, 0,
                           0, 0, 0,
                           1, 1, 1,
                           2, 2, 2, 2]
        self.velocity_plot = velocity_plot
        
    def add(self, target):
        if self.velocity_plot:
            # state = np.concatenate((self.env.state[1:6:2].flatten(), target[1:6:2], self.env.ang.flatten(), self.env.clipped_action.flatten()))
            state = np.concatenate((self.env.state[1:6:2].flatten(), target[1:6:2], self.env.ang.flatten(), (self.env.step_effort.flatten()+1)*T2WR*M*G/8 ))
        else:
            state = np.concatenate((self.env.state[0:5:2].flatten(), self.env.ang.flatten(), self.env.clipped_action.flatten()))
        self.states.append(state)
        self.times.append(self.env.i*self.env.t_step)

        
    def clear(self,):
        self.states = []
        self.times = []
        
    def plot(self, nome='padrao'):
        P = 0.7
        fig, self.axs = plt.subplots(3, figsize = (7, 7*1.414), dpi=300)
        # fig.suptitle(nome)
        self.states = np.array(self.states)
        self.times = np.array(self.times)
        for print_state, label, line_style, axis_place, color, name in zip(self.print_list, self.plot_labels, self.line_styles, self.plot_place, self.color, self.axis_labels):
            self.axs[axis_place].plot(self.times, self.states[:,print_state], label = label, ls=line_style, lw=0.8, color = color)
            self.axs[axis_place].legend()
            self.axs[axis_place].grid(True)
            self.axs[axis_place].set(ylabel=name)
        self.axs[2].axhline(y=T2WR*M*G/4)
        self.axs[2].axhline(y=0)
        plt.xlabel('tempo (s)')
        plt.savefig(nome+'.pgf', bbox_inches='tight')
        plt.savefig(nome+'.png', bbox_inches='tight')
        # plt.title(nome[-40:])
        # plt.show()
        if self.depth_plot:
            fig3d = plt.figure('3D map')
            ax = Axes3D(fig3d)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')       
            ax.set_zlabel('z (m)')    
            states = np.array(self.states)
            times = np.array(self.times)
            xs = self.states[:,0]
            ys = self.states[:,1]
            zs = self.states[:,2]
            t = self.times
            ax.scatter(xs,ys,zs,c=plt.cm.jet(t/max(t)))
            ax.plot3D(xs,ys,zs,linewidth=0.5)
            ax.set_xlim(-BB_POS, BB_POS)
            ax.set_ylim(-BB_POS, BB_POS)
            ax.set_zlim(-BB_POS, BB_POS)
            plt.grid(True)
            # plt.show()
            
        self.clear()
