import numpy as np

def visual_reward(total_steps, marker_position, quad_position, quad_vel, control, last_shaping, step):
    done = False
    error_p = 100
    control_p = 3
    vel_p = 0
    cascading_error = [0.1, 0.5, 1]
    cascading_rew = [200, 100, 10]
    marker_position = marker_position-np.array([0, 0, 5])
    
    error_xy = np.linalg.norm(marker_position[0:2]-quad_position[0:2])
    error_z = np.abs((marker_position[2]-quad_position[2]))
    error = np.linalg.norm(marker_position-quad_position)
    
    vel = np.linalg.norm(quad_vel)
    control_effort = np.linalg.norm(control)/np.linalg.norm([1, 1, 4])
    
    cas_shap = 0
    for cas_err, cas_rew in zip(cascading_error, cascading_rew):
        if error < cas_err:
            # print(error_xy, error_z)
            # print(cas_rew)
            cas_shap = cas_rew
            break

    done_shape = 0 
    # if (error+vel)/2 < np.linalg.norm(np.ones(6)*0.001):
    #     done_shape = 500
    #     done = True

            
    # print('Error Debug')
    # print(done_shape)
    # print(cas_shap)
    # print(quad_position, )
    # print(error_p*error/(np.sqrt(3)*10))
    # print(control_p*control_effort)
    # print(np.linalg.norm(quad_vel)*vel_p)
    
    current_shaping = done_shape + cas_shap - error_p*(error) - control_p*control_effort - np.linalg.norm(quad_vel)*vel_p
   
    soft_landed = True if np.linalg.norm(quad_vel) < np.linalg.norm(np.ones(3)*0.20) else False    
    landed = True if quad_position[2] <= -4.95 else False
    on_target = True if landed and error < 0.1 else False
    astray = True if error_xy > np.sqrt(2)*3 or error_z > 7 else False
    max_steps = True if step > total_steps else False
    
    if landed:
        if soft_landed:
            current_shaping += 50
            if on_target:
                current_shaping += 200
        else:
            current_shaping -= 20
        done = True
    elif astray:
        current_shaping -= 10
        # current_shaping -= 0
        done = True
    if max_steps:
        done = True

    if last_shaping:
        reward = current_shaping - last_shaping
        # reward = current_shaping
    else:
        reward = - control_p*control_effort 
        # reward = current_shaping

    # print(reward)  
    return reward, current_shaping, done