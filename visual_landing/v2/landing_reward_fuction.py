import numpy as np
MAX_STEPS = 2000

def visual_reward(marker_position, quad_position, quad_vel, control, last_shaping, step):

    error_p = 100
    control_p = 0.3
    vel_p = 1
    cascading_error = [0.1, 0.2, 0.3]
    cascading_rew = [5, 3, 1]
    marker_position = marker_position-np.array([0, 0, 5])
    
    error = np.linalg.norm(marker_position-quad_position)
    vel = np.linalg.norm(quad_vel)
    control_effort = np.linalg.norm(control)/np.linalg.norm([1, 1, 4])
    
    for cas_err, cas_rew in zip(cascading_error, cascading_rew):
        if error < cas_err:
            cas_shap = cas_rew
            break
        else:
            cas_shap = 0
    
    if (error+vel)/2 < np.linalg.norm(np.ones(6)*0.001):
        done_shape = 20
        done = True
    else:
        done_shape = 0
        done = False
            
    current_shaping = done_shape + cas_shap - error_p*error/(np.sqrt(3)*10) - control_p*control_effort - np.linalg.norm(quad_vel)*vel_p
   
    soft_landed = True if np.linalg.norm(quad_vel) > np.linalg.norm(np.ones(3)*0.25) else False    
    landed = True if quad_position[2] <= -4.95 else False
    on_target = True if landed and error < 0.1 else False
    astray = True if error > np.sqrt(3)*10 else False
    max_steps = True if step > MAX_STEPS else False
    
    if landed:
        if soft_landed:
            current_shaping += 2
            if on_target:
                current_shaping += 20
        else:
            current_shaping -= 10
        done = True
    elif astray:
        current_shaping -= 10
        done = True
    if max_steps:
        done = True

    if last_shaping:
        reward = current_shaping - last_shaping
    else:
        reward = - control_p*control_effort 

        
    return reward, current_shaping, done