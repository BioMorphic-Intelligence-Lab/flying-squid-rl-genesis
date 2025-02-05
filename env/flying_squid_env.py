import genesis as gs
import gymnasium as gym
import numpy as np

from collections import deque
from stable_baselines3.common.vec_env import VecEnv

class FlyingSquidEnv(VecEnv):
    def __init__(self, vis=False, device='cpu',
                 max_steps=2000,
                 max_lin_vel=1.5, max_rot_vel=2,
                 history_length=100,
                 num_envs=10,
                 dt=0.01, g=9.81,
                 corridor_width_range=[2, 5],
                 corridor_angle_range=np.deg2rad([-45, 45]),
                 p_ini=None,
                 yaw_ini=None):

        # Init Genesis running on CPU
        if str.lower(device) == 'cpu':
            gs.init(backend=gs.cpu, precision="32", logging_level='warning')
        else:
            print("ERROR! Current no other device than CPU supported")
        self.vis = vis

        # Action Space: [theta, ||v||, omega] aka
        #    - angle and magnitude of desired linear speed in body frame
        #    - desired angular velocity
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        # Observation Space { 'des_dir': [sin(phi), cos(phi)]
        #                    'contacts': [[FR, FL, RL, RR] * history_length]
        #                         'att': [[sin(theta), cos(theta)] * hisotry_length]  } 
        observation_space = gym.spaces.Dict({
             'des_dir': gym.spaces.Box(low=-1, high=1, shape=(2,)),
            'contacts': gym.spaces.MultiBinary((history_length, 4)),
                 'att': gym.spaces.Box(low=-1, high=1, shape=(history_length, 2))
        })

        # Init the Vector Env
        super().__init__(num_envs,
                         observation_space,
                         action_space)

        # Init indeces
        self.envs_idx = np.arange(self.num_envs)

        # Remember Env Configuration
        self.MAX_STEPS = max_steps
        self.MAX_LIN_VEL = max_lin_vel
        self.MAX_ROT_VEL = max_rot_vel
        self.HISTORY_LENGTH = history_length
        self.CORRIDOR_WIDTH_RANGE = corridor_width_range
        self.CORRIDOR_ANGLE_RANGE = corridor_angle_range
        self.CORRIDOR_BOX_SIZE = np.array([0.1, 50, 2])

        if p_ini is None:
            num_per_side = int(np.ceil(np.sqrt(self.num_envs)))  # Number of rows/cols
            x_idx, y_idx = np.meshgrid(range(num_per_side), range(num_per_side))
            
            # Select the first num_envs positions
            grid_positions = np.stack([x_idx.ravel(), y_idx.ravel()], axis=1)[:self.num_envs]
            
            # Center the grid at the origin
            grid_positions = (grid_positions - np.mean(grid_positions, axis=0)) * 1.5 * max(self.CORRIDOR_BOX_SIZE)
            self.P_INI = np.column_stack([grid_positions, np.ones(self.num_envs) * 1.0])  # Add Z coordinate
        else:
            self.P_INI = p_ini

        if yaw_ini is None:
            self.YAW_INI = np.array([[0] for _ in range(self.num_envs)])
        else:
            self.YAW_INI = yaw_ini
        self.g = g

        # Init step counter to terminate envs
        self.step_counts = np.zeros(self.num_envs)

        # Init Observation hist
        self.contact_hist = [deque(np.zeros([self.HISTORY_LENGTH, 4])) for _ in range(self.num_envs)]
        self.att_hist = [deque(np.zeros([self.HISTORY_LENGTH, 2])) for _ in range(self.num_envs)]

        self.scene = gs.Scene(
            viewer_options=
            gs.options.ViewerOptions(
                camera_pos=(-2.0, -20.0, 5),
                camera_lookat=(3.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=dt),
            show_viewer=vis
        )

        # Init Cam 
        self.cam = self.scene.add_camera(
                res    = (1280, 960),
                pos    = (-2.0, -20.0, 5),
                lookat = (3.0, 0.0, 0.5),
                fov    = 30,
                GUI    = False
            )

        self.ground_plane = self.scene.add_entity(gs.morphs.Plane())

        # Make corridor
        self.corridor_plane_left = self.scene.add_entity(gs.morphs.Box(size=self.CORRIDOR_BOX_SIZE, fixed=True))
        self.corridor_plane_right = self.scene.add_entity(gs.morphs.Box(size=self.CORRIDOR_BOX_SIZE, fixed=True))
        self.corridor_widths = np.random.uniform(low=self.CORRIDOR_WIDTH_RANGE[0], high=self.CORRIDOR_WIDTH_RANGE[1], size=self.num_envs)
        self.corridor_angles = np.random.uniform(low=self.CORRIDOR_ANGLE_RANGE[0], high=self.CORRIDOR_ANGLE_RANGE[1], size=self.num_envs)

        # Init desired direction
        theta = self.corridor_angles + np.random.uniform(low=-np.deg2rad(45), high=np.deg2rad(45), size=self.num_envs)
        self.des_dir = np.array([np.sin(theta), np.cos(theta)]).T
        
        # Init Genesis Scene

        self.drone = self.scene.add_entity(
            gs.morphs.URDF(
                file="./assets/drone_w_arms.urdf",  # Path to your URDF file
                fixed=True,
                merge_fixed_links=False
            )
        )

        # Arm parameters
        self.K = 0.5 * np.eye(10 * 4)
        self.MAX_TENSION = 0.9
        self.RADIUS = 0.2

        # Finally build the scene
        self.scene.build(n_envs=self.num_envs)

        # Set the corridor positions
        self._generate_corridor(width=self.corridor_widths, angle=self.corridor_angles, env_idx=self.envs_idx)

        # Get overall mass
        self.M = self.drone.get_mass()
        init_pos = np.concatenate([self.P_INI, self.YAW_INI, np.zeros([self.num_envs, 10 * 4])], axis=1) 

        self.drone.set_dofs_position(init_pos)
        self.drone.set_dofs_velocity(np.zeros_like(init_pos))

    def _generate_corridor(self, width, angle, env_idx):
        # Ensure `angle` is a column vector of shape (len(env_idx), 1) for broadcasting
        cos_a = np.cos(angle)[:, np.newaxis]
        sin_a = np.sin(angle)[:, np.newaxis]
        zeros_a = np.zeros_like(angle)[:, np.newaxis]
        ones_a = np.ones_like(angle)[:, np.newaxis]

        # Make `rot` shape (len(env_idx), 3, 3)
        rot = np.stack([
            np.concatenate([cos_a, -sin_a, zeros_a], axis=1),
            np.concatenate([sin_a, cos_a, zeros_a], axis=1),
            np.concatenate([zeros_a, zeros_a, ones_a], axis=1)
        ], axis=1)  # Shape: (len(env_idx), 3, 3)

        # Broadcastable translation components
        translation = self.P_INI[env_idx, :] * [1, 1, 0] - [0, 0.5, 0]

        # Define vectors with correct shape
        box_offset = np.array([self.CORRIDOR_BOX_SIZE[0], self.CORRIDOR_BOX_SIZE[1], self.CORRIDOR_BOX_SIZE[2]])
        vec1 = np.stack([-width - box_offset[0], ones_a[:, 0] * box_offset[1], ones_a[:, 0] * box_offset[2]], axis=1)
        vec2 = np.stack([ width + box_offset[0], ones_a[:, 0] * box_offset[1], ones_a[:, 0] * box_offset[2]], axis=1)

        # Apply rotation for each environment instance
        p1 = translation + np.einsum('bij,bj->bi', rot, vec1) * 0.5
        p2 = translation + np.einsum('bij,bj->bi', rot, vec2) * 0.5

        # Ensure shape is (len(env_idx), 3)
        p1 = p1.reshape(len(env_idx), 3)
        p2 = p2.reshape(len(env_idx), 3)

        # Ensure `euler` has shape (len(env_idx), 3)
        euler = np.stack([
            np.zeros_like(angle),   # X rotation (zero)
            np.zeros_like(angle),   # Y rotation (zero)
            np.rad2deg(angle)       # Z rotation (converted to degrees)
        ], axis=1)  # Shape: (len(env_idx), 3)

        # Convert to quaternion
        quat = gs.utils.geom.xyz_to_quat(euler)
        
        self.corridor_plane_left.set_pos(p1, envs_idx=env_idx)
        self.corridor_plane_left.set_quat(quat, envs_idx=env_idx)

        self.corridor_plane_right.set_pos(p2, envs_idx=env_idx)
        self.corridor_plane_right.set_quat(quat, envs_idx=env_idx)

    def reset_(self, dones):
        
        num_resets = dones.sum()

        if num_resets > 0:
            # Reset obs history
            for i in range(self.num_envs):
                if dones[i]:
                    self.contact_hist[i] = deque(np.zeros([self.HISTORY_LENGTH, 4]))
                    self.att_hist[i] = deque(np.zeros([self.HISTORY_LENGTH, 2]))

            # Make corridor
            self.corridor_widths[dones] = np.random.uniform(low=self.CORRIDOR_WIDTH_RANGE[0], high=self.CORRIDOR_WIDTH_RANGE[1], size=num_resets)
            self.corridor_angles[dones] = np.random.uniform(low=self.CORRIDOR_ANGLE_RANGE[0], high=self.CORRIDOR_ANGLE_RANGE[1], size=num_resets)
            self._generate_corridor(width=self.corridor_widths, angle=self.corridor_angles, env_idx=self.envs_idx[dones])

            # Init desired direction
            theta = self.corridor_angles[dones] + np.random.uniform(low=-np.deg2rad(45), high=np.deg2rad(45), size=num_resets)
            self.des_dir[dones, :] = np.array([np.sin(theta), np.cos(theta)]).T

            # Reset Drone Positions
            init_pos = np.concatenate([self.P_INI[dones], self.YAW_INI[dones], np.zeros([num_resets, 10 * 4])], axis=1) 
            self.drone.set_dofs_position(init_pos, envs_idx=self.envs_idx[dones])
            self.drone.set_dofs_velocity(np.zeros_like(init_pos), envs_idx=self.envs_idx[dones])
            self.step_counts[dones] = np.zeros(num_resets) 

        return self._get_observation()

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.actions = actions

    def _lin_vel_ctrl(self, v, v_des,kp=10):
        force = np.array([[0, 0, self.g * self.M] for _ in range(self.num_envs)])
        force[:, :2] += kp * (v_des[:, :2] - np.array(v[:, :2]))

        return force

    def _rot_vel_ctrl(self, omega, omega_des, kp=10):
        torque = kp * (omega_des - np.array(omega))
        return torque

    def _arm_ctrl(self, alpha):
        alpha = np.clip(alpha, a_min=-1, a_max=1)
        tension = alpha * self.MAX_TENSION
        return np.ones([self.num_envs, 10 * 4]) * self.RADIUS * tension

    def step_wait(self):

        self.step_counts += 1

        self.actions = np.clip(
            self.actions * [np.pi, self.MAX_LIN_VEL, self.MAX_ROT_VEL],
            a_min=[-np.pi, 0, -self.MAX_ROT_VEL],
            a_max=[np.pi, self.MAX_LIN_VEL, self.MAX_ROT_VEL]
        ).reshape([self.num_envs, 3])  

        # Find current state
        p = self.drone.get_dofs_position()
        v = self.drone.get_dofs_velocity()

        # Extract the commands from the actions
        v_des = self.actions[:, 1, np.newaxis] * np.hstack([np.sin(self.actions[:, 0]), np.cos(self.actions[:, 0])]) * self.MAX_LIN_VEL
        omega_des = self.actions[:, 2] * self.MAX_ROT_VEL
        
        # Apply torque within limits
        lin_ctrl = self._lin_vel_ctrl(v=v[:, :3], v_des=v_des)
        rot_ctrl = self._rot_vel_ctrl(omega=v[:, 3], omega_des=omega_des).reshape([self.num_envs, 1])
        arm_ctrl = self._arm_ctrl(alpha=0.8)

        # Find stiffness contribution for the arm
        stiffness_contrib = -self.K @ np.array(p[:, 4:]).T

        # Apply the control and stiffness forces
        self.drone.control_dofs_force(
            np.concatenate([lin_ctrl, rot_ctrl, arm_ctrl + stiffness_contrib.T], axis=1)
        )
        self.scene.step()

        # Episode ends if the pendulum falls
        max_steps_reached = self.step_counts > self.MAX_STEPS
        dones = (max_steps_reached)

        # Compute reward
        distance_traveled = np.dot(self.des_dir.T, p[:, :2])      # Encourage distance traveled along the desired direction
        rewards = (distance_traveled)

        # Update Observation hist
        contacts = 1.0 * (np.linalg.norm(self.drone.get_links_net_contact_force()[:, -4:, :], axis=2) > 0)
        for n in range(self.num_envs):
            self.contact_hist[n].popleft()
            self.contact_hist[n].append(contacts[n, :])
            self.att_hist[n].popleft()
            self.att_hist[n].append(np.array([np.sin(p[n, 3]), np.cos(p[n, 3])]))
        
        # Write info dicts
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self._get_observation()
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True

        # Reset done environments
        self.reset_(dones=dones)

        return self._get_observation(), rewards, dones, infos
    
    def _get_observation(self):        

        contact_hist = np.concatenate([list(contact) for contact in self.contact_hist], axis=1)
        att_hist = np.concatenate([list(att) for att in self.att_hist], axis=1)
        return { 'des_dir': self.des_dir,
                'contacts': contact_hist,
                     'att': att_hist}
    
    def close(self):
        pass
    
    def seed(self):
        pass

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None for _ in range(self.num_envs)]
    
    def set_attr(self, attr_name, value, indices=None):
        pass
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]*self.num_envs