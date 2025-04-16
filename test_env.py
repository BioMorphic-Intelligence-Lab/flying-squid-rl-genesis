import argparse
import numpy as np
import genesis as gs
import matplotlib.pyplot as plt
from baseline.baseline import Baseline
from stable_baselines3 import PPO
from env import FlyingSquidEnv

def read_po():
    parser = argparse.ArgumentParser(description="Simulation of the flying squid.")
    parser.add_argument('--vis', action='store_true', help="Flag on whether the simulation is visualized.")
    parser.add_argument('--dt', type=float, default=0.01, help="Simulation step size")
    parser.add_argument('--T', type=float, default=25.0, help="Simulation end time")
    parser.add_argument('--record', action='store_true', help="Record experiment to video")
    parser.add_argument('--plot', action='store_true', help="Whether or not to plot the trial.")
    parser.add_argument('--n_envs', type=int, default=1, help="Number of parallel environmnents")
    parser.add_argument('--debug', action='store_true', help="Wether or not to draw debug arrows")
    parser.add_argument('--corridor', action='store_true', help="Wether or not to make a corridor")
    parser.add_argument('--obstacle_density', type=float, default=0.025, help="Obstacle density")
    return parser.parse_args()

def main():
    args = read_po()
    n_steps = int(args.T/args.dt)
    env = FlyingSquidEnv(num_envs=args.n_envs, vis=args.vis, max_steps=n_steps,
                         obstacle_density=args.obstacle_density, corridor=args.corridor,
                         dt=args.dt, history_duration=15.0, observation_length=10,
                         debug=args.debug)

    model = PPO.load("./models/named_models/best_follower.zip")
    bl = Baseline()

    obs = env.reset()

    if args.plot:
        p_gt = np.zeros([n_steps, env.num_envs, 3])
        v_gt = np.zeros([n_steps, env.num_envs, 3]) 
        omega_gt = np.zeros([n_steps, env.num_envs, 1])
        v_des = np.zeros([n_steps, env.num_envs, 3])
        omega_des = np.zeros([n_steps, env.num_envs, 1])

    a = np.zeros([env.num_envs, 3])
    acc_reward = 0

    if args.record:
        env.cam.start_recording()

    for t in range(n_steps):
        for j in range(env.num_envs):
            obs_j = {key: value[j] for key, value in obs.items()}
            a[j, :] = bl.act(obs_j)
            #a[j, :] = [0.5, # theta / np.pi
            #           1.0, # ||v|| / MAX_SPEED
            #           0.1] # omega / MAX_RATE
            #a[j, :], _ = model.predict(obs_j)

        obs, rewards, dones, infos = env.step(a)

        acc_reward += rewards
        print(f"Accumulative Reward: {acc_reward}")

        if args.plot:
            p_gt[t, :, :] = env.drone.get_dofs_position()[:, :3]
            v_gt[t, :, :] = env.drone.get_dofs_velocity()[:, :3]
            omega_gt[t, :, :] = env.drone.get_dofs_velocity()[:, 3]
            angle = np.arctan2(obs['des_dir'][0][1], obs['des_dir'][0][0]) - a[0, 0] * np.pi
            v_des[t, :, :] = env.MAX_LIN_VEL * (1 - a[0, 1]) * np.array([np.cos(angle), np.sin(angle), 0.0])
            omega_des[t, :, :] = env.MAX_ROT_VEL * a[0, 2]

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


    if args.plot:
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))
        #axs[0].plot(np.arange(n_steps) * args.dt, p_gt[:, :, 0])
        axs[0].plot(np.arange(n_steps) * args.dt, v_gt[:, :, 0], "--")
        axs[0].plot(np.arange(n_steps) * args.dt, v_des[:, :, 0], ":", color="black")
        #axs[1].plot(np.arange(n_steps) * args.dt, p_gt[:, :, 1])
        axs[1].plot(np.arange(n_steps) * args.dt, v_gt[:, :, 1], "--")
        axs[1].plot(np.arange(n_steps) * args.dt, v_des[:, :, 1], ":", color="black")
        #axs[2].plot(np.arange(n_steps) * args.dt, p_gt[:, :, 2])
        axs[2].plot(np.arange(n_steps) * args.dt, v_gt[:, :, 2], "--")
        axs[2].plot(np.arange(n_steps) * args.dt, v_des[:, :, 2], ":", color="black")

        axs[3].plot(np.arange(n_steps) * args.dt, omega_gt[:, :, 0], "--")
        axs[3].plot(np.arange(n_steps) * args.dt, omega_des[:, :, 0], ":", color="black")

        plt.show()

if __name__ == "__main__":
    main()