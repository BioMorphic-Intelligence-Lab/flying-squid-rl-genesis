import os
import sys
import torch
from env import FlyingSquidEnv
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import VecMonitor

def main(argv):

    models_dir = "models/PPO"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    env = VecMonitor(
        FlyingSquidEnv(num_envs=50, max_steps=3000, corridor=True, obstacle_density=0.0,
                       dt=0.01, history_length=100)
    )

    if "-c" in argv:
        argument = argv[argv.index("-c") + 1]
        if argument.isdigit():
            number = int(argument)
            model = PPO.load(f"models/PPO/{number}.0")  
            model.set_env(env)
        else:
            number = 0
            model = PPO.load(argument)  
            model.set_env(env)

        params = model.get_parameters()
        # Modify the optimizer state
        if "policy.optimizer" in params:
            for param_group in params["policy.optimizer"]["param_groups"]:
                param_group["lr"] = 1e-20  # Set your desired learning rate
                param_group["entropy_coeff"] = 1e-4
                param_group["gamma"] = 1.0 - 1e-3
        model.set_parameters(params)


    else:
        number = 0
        net_arch_dict = dict({"pi": [64, 64, 64, 64], "vf": [64, 64, 64, 64]})

        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                            net_arch=net_arch_dict,
                            log_std_init = 0,
                            ortho_init=True)
        
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir,
                     policy_kwargs=policy_kwargs, learning_rate=1e-10)
    
    # Train the agent
    TIMESTEPS = 1e6
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name="PPO",
                progress_bar=True
                )
        model.save(f"{models_dir}/{number + iters*TIMESTEPS}")

if __name__ == "__main__":
    main(sys.argv)