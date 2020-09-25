import numpy as np

m_c = [1,
       1,
       1,
       1,
       1,
       1,
       1,
       3,
       3,
       3,
       3,
       3,
       3,
       3]

time = [1,
        0.01,
        0.01,
        1,
        2,
        2,
        8,
        0,
        0,
        0,
        0,
        0,
        0,
        0]

target = [[0,0,0],
          [1, 0, 0],
          [0, 0, 1],
          [1, 1, 1],
          [1, -2, 3],
          [1, 1, -2],
          [10, 10, 10],
          [4000, 5000, 2, np.pi/10, 0.3, np.array([0,0,0])],
          [4000, 5000, 1, np.pi/10, 0.3, np.array([0,0,0])],
          [4000, 5000, 1, np.pi/10, 2, np.array([0,0,0])],
          [4000, 5000, 1, np.pi/4, 0.3, np.array([0,0,0])],
          [4000, 5000, 1, np.pi/3, 0.3, np.array([0,0,0])],
          [4000, 5000, 1, np.pi/3, 1, np.array([0,0,0])],
          [4000, 5000, 0.5, np.pi/3, 1, np.array([0,0,0])]]

def target_parse(n_episode):
    return m_c[n_episode], time[n_episode], target[n_episode]

def episode_n():
    return len(m_c)