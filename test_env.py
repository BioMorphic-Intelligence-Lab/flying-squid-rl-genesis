import argparse

import numpy as np
import genesis as gs
from baseline.baseline import Baseline
from stable_baselines3 import PPO
from env import FlyingSquidEnv

def read_po():
    parser = argparse.ArgumentParser(description="Simulation of the flying squid.")
    parser.add_argument('--vis', action='store_true', help="Flag on whether the simulation is visualized.")
    parser.add_argument('--dt', type=float, default=0.01, help="Simulation step size")
    parser.add_argument('--T', type=float, default=25.0, help="Simulation end time")
    parser.add_argument('--record', action='store_true', help="Record experiment to video")
    parser.add_argument('--n_envs', type=int, default=1, help="Number of parallel environmnents")
    parser.add_argument('--debug', action='store_true', help="Wether or not to draw debug arrows")
    return parser.parse_args()

def main():
    args = read_po()

    #model = PPO.load("models/PPO/5000000.0")
    baseline = Baseline()

    n_steps = int(args.T/args.dt)
    env = FlyingSquidEnv(num_envs=args.n_envs, vis=args.vis, max_steps=int(n_steps/3),
                        dt=args.dt, history_length=100, debug=args.debug)
    obs = env.reset()

    a = np.zeros([env.num_envs, 3])

    if args.record:
        env.cam.start_recording()

    for t in range(n_steps):
        for j in range(env.num_envs):
            obs_j = {key: value[j] for key, value in obs.items()}
            a[j, :] = baseline.act(obs_j)
            #a[j, :] = [0.0, # \delta theta / np.pi
            #           0.0, # \delta||v|| / MAX_SPEED
            #           0.0] # omega / MAX_RATE
            #a[j, :], _ = model.predict(obs_j)
            
        obs, rewards, dones, infos = env.step(a)

        if np.isnan(env.drone.get_dofs_position()[:3]).any():
            print("NaN detected in position. Exiting.")
            break

        if args.record and t % int(1.0 / (24.0 * args.dt)) == 0:
            squid_pos = np.array(env.drone.get_dofs_position())[0, :3]
            env.cam.set_pose(pos=squid_pos + np.array([-0, -10, 10]),
                             lookat=squid_pos)
            env.cam.render()

    if args.record:
        env.cam.stop_recording(save_to_filename='video.mp4', fps=24)

if __name__ == "__main__":
    main()