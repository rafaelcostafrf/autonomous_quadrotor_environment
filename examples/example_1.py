import numpy as np
from environment.quadrotor_env import quad, plotter

t_step = 0.01
n = 2000
quad = quad(t_step, n, direct_control=1)
quad_plot = plotter(quad, depth_plot=True)
states, _ = quad.reset()
quad_plot.add()
done = False

while not done:
    action = np.array([0, 0, 0, 0])
    states, _, done = quad.step(action)
    quad_plot.add()
quad_plot.plot()