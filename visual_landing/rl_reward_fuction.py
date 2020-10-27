import numpy as np
EPS = 1e-5
def visual_reward(total_steps, marker_position, quad_position, quad_vel, control, last_shaping, step, ang):
    done = False
    error_p = 6
    control_p = 3
    vel_p = 2
    reward = 0
    cascading_error = [0.3, 1, 2, 5]
    cascading_rew = [80, 40, 20, 10]
    marker_position = marker_position-np.array([0, 0, 5])
    
    error_xy = np.linalg.norm(marker_position[0:2]-quad_position[0:2])
    error_z = np.abs((marker_position[2]-quad_position[2]))
    error = np.linalg.norm(marker_position-quad_position)
    vel = np.linalg.norm(quad_vel)
    control_effort = np.linalg.norm(control)
    
    # for cas_err, cas_rew in zip(cascading_error, cascading_rew):
    #     if np.sqrt(error**2+vel**2) < cas_err:
    #         reward += cas_rew

    # done_shape = 0 
    # if (error+vel)/2 < np.linalg.norm(np.ones(6)*0.001):
    #     reward += 500
    #     done = True

            
    # print('Error Debug')
    # print(done_shape)
    # print(cas_shap)
    # print(quad_position, )
    # print(error_p*error/(np.sqrt(3)*10))
    # print(control_p*control_effort)
    # print(np.linalg.norm(quad_vel)*vel_p)
    
    current_shaping = - 100*(error_p*(error_z)/7 + error_p*3*error_xy/np.linalg.norm([3, 3]) + vel_p*np.linalg.norm(quad_vel)/np.linalg.norm([1,1,2])) 
   
    soft_landed = True if np.linalg.norm(quad_vel) < np.linalg.norm(np.ones(3)*0.20) else False    
    landed = True if quad_position[2] <= -4.95 else False
    on_target = True if landed and error < 0.1 else False
    astray = True if error_xy > error_z/7*4+0.7 or error_z > 7 else False
    max_steps = True if step > total_steps else False
    unstable = True if np.linalg.norm(ang) > np.linalg.norm(np.ones(3)*np.pi/4) else False
    if last_shaping:
        reward += current_shaping - last_shaping - control_p*control_effort 
        # reward = current_shaping - control_p*control_effort - vel_p*np.linalg.norm(quad_vel)/np.linalg.norm([1,1,2])
    else:
        reward += - control_p*control_effort
        # reward = current_shaping - control_p*control_effort - vel_p*np.linalg.norm(quad_vel)/np.linalg.norm([1,1,2])
    
    if unstable:
        done = True
        reward = -500
    else:
        if landed:
            if soft_landed:
                reward = 40/(error+EPS)
            else:
                reward += -100
            done = True
        elif astray:
            reward = -200
            # current_shaping -= 0
            done = True
        if max_steps:
            done = True

    

    # print(reward)  
    return reward, current_shaping, done