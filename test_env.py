import numpy as np
import genesis as gs
from stable_baselines3 import PPO
from env import FlyingSquidEnv

env = FlyingSquidEnv(num_envs=2, vis=True, max_steps=500,
                     dt=0.01, history_length=5)

a = np.zeros([env.num_envs, 3])

for i in range(1500):

    for j in range(env.num_envs):
        a[j, :] = [0.0, # theta / np.pi
                   0.5, # ||v|| / MAX_SPEED
                   0.0] # omega / MAX_RATE
    obs, rewards, dones, infos = env.step(a)