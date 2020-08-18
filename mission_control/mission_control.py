import numpy as np

class mission():
    def __init__(self, time_step):
        self.time_step = time_step
        
    def gen_trajectory(self, steps, position, velocity=None, additive=None):
        self.trajectory_step = 0
        self.trajectory_total_steps = steps
        if additive is None:
            initial_state = np.zeros(14)
        else:
            initial_state = additive
        
        self.trajectory = np.zeros([steps, 3])
        self.velocity = np.zeros([steps, 3])
        
        if velocity is None:
            for i in range(3):
                self.trajectory[:, i] = np.linspace(initial_state[i], position[i]+initial_state[i], steps)
        else:
            for i in range(3):
                self.velocity[:, i] = np.linspace(0, velocity[i], steps)
            for i in range(steps-1):
                for j in range(3):
                    self.trajectory[i+1, j] = self.trajectory[i, j] + self.velocity[i, j]*self.time_step

    def sin_trajectory(self, steps, circular_rate, ascent_rate, center, axis):
        self.trajectory_step = 0
        self.trajectory_total_steps = steps
        self.trajectory = np.zeros([steps, 3])
        self.velocity = np.zeros([steps, 3])
        self.trajectory_timesteps = np.arange(0, steps, 1)
        for step in self.trajectory_timesteps:
            a = step*circular_rate*self.time_step
            self.trajectory[step, :] = center + np.sin(a) * axis
            self.velocity[step, :] = np.cos(a) * circular_rate * axis
            self.trajectory[step, 2] = self.trajectory[step-1, 2] + ascent_rate*self.time_step
            self.velocity[step, 2] = ascent_rate
        print(self.trajectory)
        print(self.velocity)
            
    def spiral_trajectory(self, steps, rate, circular_rate, radius, center):
        self.trajectory_step = 0
        self.trajectory_total_steps = steps
        self.trajectory = np.zeros([steps, 3])
        self.velocity = np.zeros([steps, 3])
        self.trajectory_timesteps = np.arange(0, steps, 1)
        for step in self.trajectory_timesteps:
            a = step*circular_rate*self.time_step
            x = np.cos(a)*radius
            y = np.sin(a)*radius
            z = step*rate*self.time_step
            self.trajectory[step, :] = center + np.array([x, y, z]) - np.array([radius, 0, 0])
            self.velocity[step, :] = np.array([-np.sin(a)*circular_rate*radius, np.cos(a)*circular_rate*radius,  rate])
            
    def get_error(self, time):
        if self.trajectory_step == self.trajectory_total_steps:
            self.trajectory[-1,:] = self.trajectory[-1,:] + self.velocity[-1,:]*self.time_step
            mission_error = np.array([self.trajectory[-1, 0], self.velocity[-1, 0],
                                      self.trajectory[-1, 1], self.velocity[-1, 1],
                                      self.trajectory[-1, 2], self.velocity[-1, 2],
                                      0, 0, 0, 0,
                                      0, 0, 0, 0])
        else:
            mission_error = np.array([self.trajectory[self.trajectory_step, 0], self.velocity[self.trajectory_step, 0],
                                      self.trajectory[self.trajectory_step, 1], self.velocity[self.trajectory_step, 1],
                                      self.trajectory[self.trajectory_step, 2], self.velocity[self.trajectory_step, 2],
                                      0, 0, 0, 0,
                                      0, 0, 0, 0])
            self.trajectory_step += 1
        return mission_error
        

# a = mission(0.01)
# a.gen_trajectory(100, np.array([1, 0, 0]), np.array([1, 0, 0]))
# print(a.velocity, a.trajectory)