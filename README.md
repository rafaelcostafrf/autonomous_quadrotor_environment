## Quadrotor Environment
# Quadrotor environment using python, with animation possibilities.

Quadrotor mathematical model from master's thesis "Modelagem Dinâmica E Controle De Um Veículo aéreo Não Tripulado Do Tipo Quadrirrotor" by Allan Carlos Ferreira De Oliveira, Brasil, São Paulo, Santo André, UFABC - 2019

Quadrotor dynamics incorporates:
	1. Full inertia matrix (if needed).
	
	2. All calculations with quaternion angular position and velocities (no gymbal lock, even with euler flag on)
	
	2. Drag forces, based on linear and angular velocity
	
	3. Gyroscopic momentum based on prop speed.

## How to Use
# quadrotor
Declare the environment:

	simple way:
		env = quad(time_step, max_timesteps, euler=1)
		where time_step is the integration timestep (normally 0.01s), max_timesteps is the maximum env steps
		and euler is a flag for outputting euler angles instead of quaternions.
		
	there are additional flags for some specific uses, as:
		direct_control = True: 
			direct control sets the input action as individual forces in each motor.
			action should be bounded in [-1,1]
			action will be converted linearly to [0, T2WR*M*G/8], 
			where T2WR is the motor thrust to weight ratio, M the quadcopter mass and G gravity.
			
		direct_control = False:
			this sets the action in [F_z, M_x, M_y, M_z] format, where
			F_z is the thrust force of the entire quadrotor in body frame
			M_x is the moment around x in the body frame
			M_y is the moment around y in the body frame
			M_z is the moment around z in the body frame
			
			This method normally is better for linear controllers, and internally bounds the forces between [0, T2WR*M*G/8]
			as the motors of a quadcopter can't shut down or work backwards.
			
			The input should still be bounded in [-1,1] and is modulated by IC_THRUST and IC_MOMENTUM.
			
			Note that F_z action is normalized: F_z = F_in*IC_THRUST + M*G
			M_x, M_y and M_z are normalized as follows: M = M_in*IC_MOMENTUM
			
			checking env.clipped_action is important, as some actions might be clipped to keep bounded motor forces.
			
			
		deep_learning:
			Deep learning mode means that each env.step returns desired deep learning input and a reward
			Deep learning input history size is set by T and includes:
				action, position, velocity, quaternion, quaternion velocity for actual time and (T-1) past states		
			Deep learning input may be changed internally to include other states.
			Reward function was set empirically, if needed, change it internally.
		
		T: history size, normally set as 1	
		
		debug: reward function debug, normally used to fine tune reward weights.
			
env.reset()

	resets the environment.
	
	aditional flag	det_state:
		resets the environment with the determined state, that should be as follows:
			if euler flag is on:
				[x, dx, y, dy, z, dz, phi, theta, psi, w_xx, w_yy, w_zz]
			if euler flag is off:
				[x, dx, y, dy, z, dz, q_0, q_1, q_2, q_3, w_xx, w_yy, w_zz]

	returns the initial state
	
env.step(action):

	moves the environment foward one timestep, actuated by given action.
	
	returns:
		if deep learning flag:
			deep learning input, reward and done
		
		elif euler flag:
			new state in euler angles, as follows:
				[x, dx, y, dy, z, dz, phi, theta, psi, w_xx, w_yy, w_zz]
			and a done statement
		else:
			new state in quaternions:
				[x, dx, y, dy, z, dz, q_0, q_1, q_2, q_3, w_xx, w_yy, w_zz]
			and a done statement
			
			
all other functions are used internally and should'nt be called outside of the environment.

# plotter

# animation
