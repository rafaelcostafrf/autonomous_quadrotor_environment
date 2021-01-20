import numpy as np
EPS = 1e-4
def visual_reward(total_steps, marker_position, quad_position, quad_vel, control, last_shaping, step, ang, v_ang):
    """
    

    Parameters
    ----------
    total_steps : int
        Total Steps of a episode, returns done if it is reached.
    marker_position : numpy array [x, y, z]
        Position of the marker in a 3D numpy array [x, y, z]
    quad_position : numpy array [x, y, z]
        Position of the quadrotor in a 3D numpy array [x, y, z]
    quad_vel : numpy array [Vx, Vy, Vz]
        Velocity of the quadrotor in a 3D numpy array [Vx, Vy, Vz]
    control : numpy array [U1, U2, U3]
        Control effort in a 3D numpy array [U1, U2, U3]
    last_shaping : float
        Last shaping is the last timestep current shaping, given by the function itself.
    step : int
        Current episode timestep
    ang : numpy array [phi, theta, psi]
        Quadrotor Euler Angles in a 3D numpy array [phi, theta, psi]

    Returns
    -------
    reward : float
        The timestep reward, given last shaping and current parameters.
    current_shaping : float
        The current shaping of the funcion, used to feed itself on the next timestep.
    done : bool
        Returns True if the episode is done./ else False 

    """
    done = False
    error_p = 4
    control_p = 0.1
    reward = 0
    cascading_error = [0.3, 1, 2, 5]
    cascading_rew = [4, 3, 2, 1]
    marker_position = marker_position-np.array([0, 0, 5])
    
    error_xy = np.linalg.norm(marker_position[0:2]-quad_position[0:2])
    error_z = np.abs((marker_position[2]-quad_position[2]))
    error = np.linalg.norm(marker_position-quad_position)
    vel = np.linalg.norm(quad_vel)
    control_effort = np.linalg.norm(control)
    
    cas_shap = 0
    # for cas_err, cas_rew in zip(cascading_error, cascading_rew):
    #     if np.sqrt(error**2+vel**2) < cas_err:
    #         cas_shap += cas_rew



                
    current_shaping = -error_p*(error_xy*2+error_z)+cas_shap
   
    soft_landed = True if np.linalg.norm(quad_vel) < np.linalg.norm(np.ones(3)*0.30) else False    
    landed = True if quad_position[2] <= -4.95 else False
    on_target = True if error_xy < 0.14 else False
    flat_landed = True if np.linalg.norm(ang[0:2]) < np.linalg.norm(np.ones(2)*0.3491) and np.linalg.norm(v_ang) < np.linalg.norm(np.ones(2)*1) else False
    astray = True if error_xy > error_z/7*5+0.2 or error_z > 7 else False
    max_steps = True if step > total_steps else False

    if last_shaping:
        reward += current_shaping - last_shaping - control_p*control_effort
        # reward = current_shaping - control_p*control_effort - vel_p*np.linalg.norm(quad_vel)/np.linalg.norm([1,1,2])
    else:
        reward += - control_p*control_effort
        # reward = current_shaping - control_p*control_effort - vel_p*np.linalg.norm(quad_vel)/np.linalg.norm([1,1,2])
    
    if landed:
        if soft_landed:
            if flat_landed:
                if on_target:
                    reward = 5
                    print('SOLVED!')
                else:
                    reward = 1
            else:
                # print(ang[0:2])
                reward = 0
        else:
            reward = -1
        done = True
    elif astray:
        reward = -5
        done = True
    if max_steps:
        reward = -2
        done = True

    


    # print(reward)  
    return reward, current_shaping, done