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