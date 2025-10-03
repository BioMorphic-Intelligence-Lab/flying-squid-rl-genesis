import argparse
import torch
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
    parser.add_argument('--cylinder_obstacle_density', type=float, default=0.025, help="Obstacle density")
    parser.add_argument('--box_obstacle_density', type=float, default=0.025, help="Obstacle density")
    return parser.parse_args()

def main():
    args = read_po()
    n_steps = int(args.T/args.dt)
    env = FlyingSquidEnv(num_envs=args.n_envs, vis=args.vis, max_steps=n_steps,
                         cylinder_obstacle_density=args.cylinder_obstacle_density,
                         box_obstacle_density=args.box_obstacle_density,
                         corridor=args.corridor,
                         dt=args.dt, history_duration=15.0, observation_length=10,
                         debug=args.debug)

    model = PPO.load("./models/named_models/best_follower")
    bl = Baseline()

    obs = env.reset()

    if args.plot:
        p_gt = torch.zeros([n_steps, env.num_envs, 3])
        v_gt = torch.zeros([n_steps, env.num_envs, 3])
        omega_gt = torch.zeros([n_steps, env.num_envs, 1])
        v_des = torch.zeros([n_steps, env.num_envs, 3])
        omega_des = torch.zeros([n_steps, env.num_envs, 1])

    a = torch.zeros([env.num_envs, 3])
    acc_reward = 0

    if args.record:
        env.cam.start_recording()

    for t in range(n_steps):
        for j in range(env.num_envs):
            obs_j = {key: value[j] for key, value in obs.items()}
            #a[j, :] = bl.act(obs_j)
            #a[j, :] = [0.5, # theta / np.pi
            #           1.0, # ||v|| / MAX_SPEED
            #           0.1] # omega / MAX_RATE
            pred, _ = model.predict(obs_j)
            a[j, :] = torch.as_tensor(pred, dtype=torch.float32)

        obs, rewards, dones, infos = env.step(a)

        acc_reward += rewards
        print(f"Accumulative Reward: {acc_reward}")

        if args.plot:
            p_gt[t, :, :] = torch.tensor(env.drone.get_dofs_position())[:, :3]
            v_gt[t, :, :] = torch.tensor(env.drone.get_dofs_velocity())[:, :3]
            omega_gt[t, :, :] = torch.tensor(env.drone.get_dofs_velocity())[:, 3].unsqueeze(-1)
            angle = torch.atan2(obs['des_dir'][0][1], obs['des_dir'][0][0]) - a[0, 0] * torch.pi
            v_des[t, :, :] = (
                env.MAX_LIN_VEL
                * (1 - a[0, 1])
                * torch.tensor([torch.cos(angle), torch.sin(angle), torch.tensor(0.0)])
            )
            omega_des[t, :, :] = (env.MAX_ROT_VEL * a[0, 2]).reshape(1, 1)

        if torch.isnan(env.drone.get_dofs_position()[:3]).any().item():
            print("NaN detected in position. Exiting.")
            break

        if args.record and t % int(1.0 / (24.0 * args.dt)) == 0:
            squid_pos = torch.tensor(env.drone.get_dofs_position())[0, :3]
            env.cam.set_pose(pos=(squid_pos + torch.tensor([0.0, -10.0, 10.0])).tolist(),
                             lookat=squid_pos.tolist())
            env.cam.render()

    if args.record:
        env.cam.stop_recording(save_to_filename='video.mp4', fps=24)


    if args.plot:
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))
        time_axis = (torch.arange(n_steps, dtype=torch.float32) * args.dt).numpy()
        #axs[0].plot(time_axis, p_gt[:, :, 0].numpy())
        axs[0].plot(time_axis, v_gt[:, :, 0].numpy(), "--")
        axs[0].plot(time_axis, v_des[:, :, 0].numpy(), ":", color="black")
        #axs[1].plot(time_axis, p_gt[:, :, 1].numpy())
        axs[1].plot(time_axis, v_gt[:, :, 1].numpy(), "--")
        axs[1].plot(time_axis, v_des[:, :, 1].numpy(), ":", color="black")
        #axs[2].plot(time_axis, p_gt[:, :, 2].numpy())
        axs[2].plot(time_axis, v_gt[:, :, 2].numpy(), "--")
        axs[2].plot(time_axis, v_des[:, :, 2].numpy(), ":", color="black")

        axs[3].plot(time_axis, omega_gt[:, :, 0].numpy(), "--")
        axs[3].plot(time_axis, omega_des[:, :, 0].numpy(), ":", color="black")

        plt.show()

if __name__ == "__main__":
    main()