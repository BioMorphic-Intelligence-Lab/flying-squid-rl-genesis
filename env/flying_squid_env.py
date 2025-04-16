import genesis as gs
import gymnasium as gym
import numpy as np

from collections import deque
from stable_baselines3.common.vec_env import VecEnv

class FlyingSquidEnv(VecEnv):
    def __init__(self, vis=False, device='cpu',
                 max_steps=2000,
                 max_lin_vel=1.5, max_rot_vel=2,
                 history_duration=20.0,
                 observation_length=10,
                 num_envs=10,
                 dt=0.01, g=9.81,
                 corridor_width_range=[2, 5],
                 corridor_angle_range=np.deg2rad([-45, 45]),
                 p_ini=None,
                 yaw_ini=None,
                 debug=False,
                 corridor=True,
                 obstacle_density=0.025):

        # Init Genesis running on CPU
        if str.lower(device) == 'cpu':
            gs.init(backend=gs.cpu, precision="32", logging_level='error')
        else:
            print("ERROR! Currently no other device than CPU supported")

        # Should we visualize the simulation?
        self.vis = vis

        # Should we draw debug arrows?
        self.debug = debug

        # Timestep
        self.dt = dt

        # Action Space: [theta, ||v||, omega] aka
        #    - angle and magnitude of desired linear speed in body frame
        #    - desired angular velocity
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        # Observation Space { 'des_dir': [sin(phi), cos(phi)]
        #                    'contacts': [[FR, FL, RL, RR] * observation_length]
        #                         'att': [[sin(theta), cos(theta)] * observation_length]  } 
        observation_space = gym.spaces.Dict({
             'des_dir': gym.spaces.Box(low=-1, high=1, shape=(2,)),
            'contacts': gym.spaces.MultiBinary((observation_length, 4)),
                 'att': gym.spaces.Box(low=-1, high=1, shape=(observation_length, 2)),
           'prev_cmds': gym.spaces.Box(low=-1, high=1, shape=(observation_length, action_space.shape[0]))
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
        self.OBSERVATION_LENTGH = observation_length
        self.HISTORY_DURATION = history_duration
        self.HISTORY_LENGTH = int(np.ceil(self.HISTORY_DURATION / self.dt))
        self.CORRIDOR_WIDTH_RANGE = corridor_width_range
        self.CORRIDOR_ANGLE_RANGE = corridor_angle_range
        self.CORRIDOR_BOX_SIZE = np.array([0.5, 50, 2])
        self.CORRIDOR = corridor
        self.MAX_OBSTACLE_DENSITY = obstacle_density
        self.OBSTACTLE_SIZE_RANGE = np.arange(start=0.3, stop=0.8, step=0.2)
        self.FLIGHT_HEIGHT=1.25

        # Init error integral for lin vel control
        self._integral_vel_error = np.zeros([self.num_envs, 2])
        self._integral_omega_error = np.zeros([self.num_envs, 1])

        if p_ini is None:
            num_per_side = int(np.ceil(np.sqrt(self.num_envs)))  # Number of rows/cols
            x_idx, y_idx = np.meshgrid(range(num_per_side), range(num_per_side))
            
            # Select the first num_envs positions
            grid_positions = np.stack([x_idx.ravel(), y_idx.ravel()], axis=1)[:self.num_envs]
            
            # Center the grid at the origin
            grid_positions = grid_positions * 2 * max(self.CORRIDOR_BOX_SIZE)
            self.P_INI = np.column_stack([grid_positions, np.zeros(self.num_envs)])  # Add Z coordinate
        else:
            self.P_INI = p_ini

        if yaw_ini is None:
            self.YAW_INI = np.array([[0] for _ in range(self.num_envs)])
        else:
            self.YAW_INI = yaw_ini
        self.g = g

        # Init step counter to terminate envs
        self.step_counts = np.zeros(self.num_envs, dtype=int)

        # Init Observation hist
        self.contact_hist = [deque(np.zeros([self.HISTORY_LENGTH, 4])) for _ in range(self.num_envs)]
        self.att_hist = [deque(np.zeros([self.HISTORY_LENGTH, 2])) for _ in range(self.num_envs)]
        self.action_hist = [deque(np.zeros([self.HISTORY_LENGTH, self.action_space.shape[0]])) for _ in range(self.num_envs)]
        self.prev_distance = np.zeros(self.num_envs)

        # Set Contact Dynamics options
        gs.options.RigidOptions(contact_resolve_time=0.001, iterations=5000)

        # Define the scene
        self.scene = gs.Scene(
            viewer_options=
            gs.options.ViewerOptions(
                camera_pos=(-1.0, -15.0, 10),
                camera_lookat=(0.0, 10.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -self.g)),
            show_viewer=vis
        )

        # Init Cam 
        self.cam = self.scene.add_camera(
                res    = (1280, 960),
                pos    = (-1.0, -15.0, 10),
                lookat = (0.0, 10.0, 0.5),
                fov    = 30,
                GUI    = False
            )

        self.ground_plane = self.scene.add_entity(gs.morphs.Plane())

        # Make corridor
        if self.CORRIDOR:
            self.corridor_plane_left = self.scene.add_entity(gs.morphs.Box(pos=(0, -100, 0),
                                                                        size=self.CORRIDOR_BOX_SIZE,
                                                                        fixed=True),
                                                            surface=gs.surfaces.Plastic(color=(0.8, 0.8, 0.8, 0.9)))
            self.corridor_plane_right = self.scene.add_entity(gs.morphs.Box(pos=(0, -100, 0),
                                                                            size=self.CORRIDOR_BOX_SIZE,
                                                                            fixed=True),
                                                            surface=gs.surfaces.Plastic(color=(0.8, 0.8, 0.8, 0.9)))
        self.corridor_widths = np.random.uniform(low=self.CORRIDOR_WIDTH_RANGE[0], high=self.CORRIDOR_WIDTH_RANGE[1], size=self.num_envs)
        self.corridor_angles = np.random.uniform(low=self.CORRIDOR_ANGLE_RANGE[0], high=self.CORRIDOR_ANGLE_RANGE[1], size=self.num_envs)

        # Init obstacle arrays
        self.obstacles = [
            [self.scene.add_entity(
                gs.morphs.Cylinder(pos=(0, -5, 0.5 * self.CORRIDOR_BOX_SIZE[2]), euler=(0, 0, 0),
                                    radius=radius, height=self.CORRIDOR_BOX_SIZE[2],
                                    fixed=True)) 
                for _ in range(int(self.MAX_OBSTACLE_DENSITY * self.CORRIDOR_WIDTH_RANGE[1] * self.CORRIDOR_BOX_SIZE[1]))]
            for radius in self.OBSTACTLE_SIZE_RANGE
        ] 
        self.obstacle_counters = np.zeros(len(self.OBSTACTLE_SIZE_RANGE))

        # Init desired direction
        theta = np.zeros(self.num_envs) #self.corridor_angles + np.random.uniform(low=-np.deg2rad(45), high=np.deg2rad(45), size=self.num_envs)
        self.des_dir = np.array([np.sin(theta), np.cos(theta)]).T
        
        # Init Genesis Scene
        self.drone = self.scene.add_entity(
            gs.morphs.URDF(
                file="./assets/drone_w_arms.urdf",  # Path to your URDF file
                fixed=True,
                merge_fixed_links=False,
            ), visualize_contact=self.debug
        )

        # Arm parameters
        self.K = 5.0 * np.eye(10 * 4)
        self.MAX_TENSION = 10.0
        self.RADIUS = 0.2

        # Drone base damping
        self.D = np.diag([2.0, 2.0, 2.0, 1.0])

        # Finally build the scene
        self.scene.build(n_envs=self.num_envs)

        # Get overall mass
        self.M = self.drone.get_mass()
        init_pos = np.concatenate([np.zeros([self.num_envs, 2]), self.FLIGHT_HEIGHT * np.ones([self.num_envs, 1]),
                                   self.YAW_INI, np.zeros([self.num_envs, 10 * 4])], axis=1) 
        self.drone.set_pos(np.concatenate([self.P_INI[:, :2], np.zeros([self.num_envs, 1])], axis=1))
        self.drone.set_dofs_position(init_pos)
        self.drone.set_dofs_velocity(np.zeros_like(init_pos))

    def _generate_obstacles(self, density, size_range, env_idx, offset=1.0):

        # Set all obstacle positons back to the original place
        for radius in self.obstacles:
            for obstacle in radius:
                pos = np.array([[0, -5, 0.5 * self.CORRIDOR_BOX_SIZE[2]] for _ in range(len(env_idx))])
                obstacle.set_pos(pos, envs_idx=env_idx)

        # Find number of obstacles
        num_obstacles = (density * self.corridor_widths[env_idx] * (self.CORRIDOR_BOX_SIZE[1] - offset)).astype(int)

        # Init pos vector
        pos = np.zeros([len(env_idx), 3])

        # Loop over environments
        for idx, n in enumerate(num_obstacles):
            # Loop over number of obstaces
            for i in range(n):
                # Randomly select radius
                radius_idx = np.random.choice(range(*size_range))
                # Define rotation matrix z-axis
                cT = np.cos(self.corridor_angles[env_idx[idx]])
                sT = np.sin(self.corridor_angles[env_idx[idx]])
                rot = np.array([[cT, -sT, 0],
                                [sT, cT, 0],
                                [0, 0, 1
                ]])
                # Get random location inside coridor
                pos = (self.P_INI[env_idx[idx], :]
                        + rot @ np.random.uniform(
                            low= [-0.5 * self.corridor_widths[env_idx[idx]] + self.OBSTACTLE_SIZE_RANGE[radius_idx], offset, 0],
                            high=[ 0.5 * self.corridor_widths[env_idx[idx]] - self.OBSTACTLE_SIZE_RANGE[radius_idx], self.CORRIDOR_BOX_SIZE[1], 0],
                            size=3)
                )
                # Set height
                pos[2] = 0.5 * self.CORRIDOR_BOX_SIZE[2]

                # Set position of obstacle
                self.obstacles[radius_idx][i].set_pos(pos[np.newaxis, :], envs_idx=[env_idx[idx]])

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
        translation = self.P_INI[env_idx, :] * [1, 1, 0]

        # Define vectors with correct shape
        box_offset = np.array([self.CORRIDOR_BOX_SIZE[0], self.CORRIDOR_BOX_SIZE[1], self.CORRIDOR_BOX_SIZE[2]]) - [0.0, 2.0, 0]
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

            # Make corridor
            self.corridor_widths[dones] = np.random.uniform(low=self.CORRIDOR_WIDTH_RANGE[0], high=self.CORRIDOR_WIDTH_RANGE[1], size=num_resets)
            self.corridor_angles[dones] = np.random.uniform(low=self.CORRIDOR_ANGLE_RANGE[0], high=self.CORRIDOR_ANGLE_RANGE[1], size=num_resets)

            if self.CORRIDOR:
                self._generate_corridor(width=self.corridor_widths[dones], angle=self.corridor_angles[dones], env_idx=self.envs_idx[dones])

            # Generate obstacles
            self._generate_obstacles(density=self.MAX_OBSTACLE_DENSITY, size_range=[0, 2], env_idx=self.envs_idx[dones])

            # Init desired direction
            theta = -self.corridor_angles[dones] + np.random.uniform(low=-np.deg2rad(45), high=np.deg2rad(45), size=num_resets)
            self.des_dir[dones, :] = np.array([np.sin(theta), np.cos(theta)]).T

            # Reset Drone Positions
            init_pos = np.concatenate([np.zeros([num_resets, 2]), self.FLIGHT_HEIGHT * np.ones([num_resets, 1]),
                                       self.YAW_INI[dones], 0.2 * np.ones([num_resets, 10 * 4])], axis=1)
            self.drone.set_dofs_position(init_pos, envs_idx=self.envs_idx[dones])
            self.drone.set_dofs_velocity(np.zeros_like(init_pos), envs_idx=self.envs_idx[dones])
            self.step_counts[dones] = np.zeros(num_resets) 

            # Reset obs history
            for i in range(self.num_envs):
                if dones[i]:
                    self.contact_hist[i] = deque(np.zeros([self.HISTORY_LENGTH, 4]))
                    self.att_hist[i] = deque(np.zeros([self.HISTORY_LENGTH, 2]))
                    self.action_hist[i] = deque(np.zeros([self.HISTORY_LENGTH, self.action_space.shape[0]]))
                    # Init last element in action hist with optimal action
                    self.action_hist[i].popleft()
                    self.action_hist[i].append([np.arctan2(self.des_dir[i, 0], self.des_dir[i, 1]), 1.0, 0.0])

        return self._get_observation()

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.actions = actions

    def _lin_vel_ctrl(self, v, v_des, kp=100, ki=25):
        vel_error = v_des[:, :2] - np.array(v[:, :2])
        self._integral_vel_error += vel_error * self.dt
        return (kp * vel_error + ki * self._integral_vel_error)

    def _altitude_ctrl(self, h, hdot, h_des, kp=100, kd=25):
        return (
            kp * (h_des - h) - kd*hdot + 
            self.M * self.g * np.ones_like(h)
        )

    def _rot_vel_ctrl(self, omega, omega_des, kp=50, ki=25):

        omega_error = (omega_des - np.array(omega)).reshape([self.num_envs, 1])
        self._integral_omega_error += omega_error * self.dt
        torque = kp * omega_error + ki * self._integral_omega_error
        return torque

    def _arm_ctrl(self, alpha):
        alpha = np.clip(alpha, a_min=-1, a_max=1)
        tension = alpha * self.MAX_TENSION
        return np.ones([self.num_envs, 10 * 4]) * self.RADIUS * tension

    def step_wait(self):

        # Find current state
        p = np.array(self.drone.get_dofs_position())
        v = np.array(self.drone.get_dofs_velocity())

        # Increment counter
        self.step_counts += 1

        if np.max(np.abs(p)).any() > 1000:  # Arbitrary large boundary
            print(f"Object out of bounds in environment {np.argwhere(np.max(np.abs(p)))}.")

        # If the simulation broke we stop it
        n_env_broken = np.isnan(p).any(axis=1).flatten()
        if n_env_broken.any():
            timestep = self.step_counts[np.argwhere(n_env_broken)].flatten()
            print(f"NaN occured in position in environment {np.argwhere(n_env_broken)}."
                  + f"At timestep {timestep}={self.dt * timestep} s.")
        
        # Ensure the NaN does not propagate 
        p[n_env_broken, :] = np.zeros([np.sum(n_env_broken), len(p[0, :])])
        v[n_env_broken, :] = np.zeros([np.sum(n_env_broken), len(v[0, :])])
        sim_break_cost = -1e3 * n_env_broken

        # Update Observation hist
        contacts = np.zeros([self.num_envs, 4])
        contacts[~n_env_broken, :] = 1.0 * (np.linalg.norm(self.drone.get_links_net_contact_force()[~n_env_broken, -4:, :], axis=2) > 0)

        for n in range(self.num_envs):
            self.contact_hist[n].popleft()
            self.contact_hist[n].append(contacts[n, :])
            self.att_hist[n].popleft()
            self.att_hist[n].append(np.array([np.sin(p[n, 3]), np.cos(p[n, 3])]))
            self.action_hist[n].popleft()
            self.action_hist[n].append(self.actions[n, :])

        # Translate velocity command action from [-1, 1] to [0, 1]
        self.actions[:, 1] = 0.5 * self.actions[:, 1] + 0.5

        # Clip actions
        self.actions = np.clip(
            self.actions * [np.pi, 1.0, self.MAX_ROT_VEL],
                     a_min=[-np.pi, 0, -self.MAX_ROT_VEL],
                     a_max=[np.pi, 1.0, self.MAX_ROT_VEL]
        ).reshape([self.num_envs, 3])  

        # Extract the commands from the actions
        # The angle is in the body frame so we rotate it accordingly
        angle = np.arctan2(np.sin(self.actions[:, 0] - p[:, 3]),
                           np.cos(self.actions[:, 0] - p[:, 3])) 
        v_des = self.actions[:, 1, np.newaxis] * self.MAX_LIN_VEL * np.vstack([np.sin(angle), np.cos(angle)]).T
        omega_des = self.actions[:, 2]

        # Draw command arrows
        if self.debug:  
            self.scene.clear_debug_objects()
            for n in range(self.num_envs):      
                # Commanded Direction
                self.scene.draw_debug_arrow(self.P_INI[n, :] + p[n, :3], np.concatenate([v_des[n, :], np.zeros(1)]),
                                        radius=0.01, color=(1, 0, 0, 0.5))
                # Commanded Angular Velocity
                self.scene.draw_debug_arrow(self.P_INI[n, :] + p[n, :3],
                                            np.concatenate([np.zeros(2), omega_des[n, np.newaxis]]),
                                            radius=0.01, color=(1, 1, 0, 0.5))
                # Desired Direction
                self.scene.draw_debug_arrow(self.P_INI[n, :] + p[n, :3], np.concatenate([self.des_dir[n, :], np.zeros(1)]),
                                            radius=0.01, color=(0, 0, 1, 0.5))
        
        # Apply torque within limits
        lin_ctrl = self._lin_vel_ctrl(v=v[:, :2], v_des=v_des)
        h_ctrl = self._altitude_ctrl(h=p[:, 2], hdot=v[:, 2], h_des=self.FLIGHT_HEIGHT).reshape([self.num_envs, 1])
        rot_ctrl = self._rot_vel_ctrl(omega=v[:, 3], omega_des=omega_des).reshape([self.num_envs, 1])
        arm_ctrl = self._arm_ctrl(alpha=0.8)

        # Find stiffness contribution for the arm
        stiffness_contrib = -self.K @ np.array(p[:, 4:]).T
        # Find damping contribution for the base
        damping_contrib = -self.D @ np.array(v[:, :4]).T

        # Apply the control, stiffness, and damping forces
        self.drone.control_dofs_force(
            np.concatenate([lin_ctrl + damping_contrib[:2, :].T,
                            h_ctrl + damping_contrib[2:3, :].T,
                            rot_ctrl + damping_contrib[3:4, :].T,
                            arm_ctrl + stiffness_contrib.T], axis=1)
        )

        # Episode ends if we reached the max number of steps
        max_steps_reached = self.step_counts >= self.MAX_STEPS
        dones = (max_steps_reached | n_env_broken)

        # Compute reward
        distance = np.sum(self.des_dir * p[:, :2], axis=1)
        distance_traveled = (distance                                          # Encourage distance traveled along
                              - self.prev_distance)                            # the desired direction
        self.prev_distance = distance
        #actuation_cost = (- 1e1 * self.actions[:, 0]**2                        # Penalize deviation from target direction
        #                  - 1e1 * self.actions[:, 1]**2                        # Penalize deviation from max vel
        #                  - 1e1 * self.actions[:, 2]**2)                       # Penalize rotational velocity
        last_action = np.array(self.action_hist)[:, -1, :]
        actuation_variation_cost = - (np.dot((last_action - self.actions)**2,  # Penalize variation of actuation, i.e weighted  
                                              1e1 * np.array([1, 1, 1])))      # sum to last action
        rewards = (1e3 * distance_traveled #+ actuation_cost
                   + actuation_variation_cost + sim_break_cost)
        
        # Write info dicts
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self._get_observation()
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True

        # Reset done environments
        self.reset_(dones=dones)

        # Step the simulator
        self.scene.step()

        # Get observation
        obs = self._get_observation()

        for key in obs:
            if np.any(np.isnan(obs[key])) or np.any(np.isinf(obs[key])):
                print(f"Warning: NaN or Inf detected in observations[{key}]!")
                print(f"p = {p[np.isnan(p)]} at index {np.argwhere(np.isnan(p))}")
        if np.isnan(rewards).any() or np.isinf(rewards).any():
            print("Warning: NaN or Inf detected in rewards!")


        return obs, rewards, dones, infos
    
    def _get_observation(self):     

        # Generate logarithmically spaced window boundaries   
        log_edges = np.logspace(0, np.log10(self.HISTORY_LENGTH), num=self.OBSERVATION_LENTGH+1, endpoint=True) 
        log_edges = self.HISTORY_LENGTH - np.round(log_edges[::-1]).astype(int)  # reverse so small windows come last

        # Ensure indices are within bounds and ensure last observation is accumulated as singleton
        log_edges[-1] = self.HISTORY_LENGTH
        log_edges[-2] = self.HISTORY_LENGTH - 1
        log_edges[0] = 0

        # Get the history of contacts, attitude, and actions as array instead of dequeues
        contact_hist = np.array([contact for contact in self.contact_hist])
        att_hist = np.array([att for att in self.att_hist])
        action_hist = np.array([action for action in self.action_hist])

        # Create empty arrays for contact, attitude, and action history observations
        contact_obs = np.zeros([self.num_envs, self.OBSERVATION_LENTGH, 4], dtype=bool)
        att_obs = np.zeros([self.num_envs, self.OBSERVATION_LENTGH, 2])
        action_obs = np.zeros([self.num_envs, self.OBSERVATION_LENTGH, self.action_space.shape[0]])

        for i in range(self.OBSERVATION_LENTGH):
            # Define start and stop indeces
            start = log_edges[i]
            end = log_edges[i+1]
            
            # Binary or accumulation of the contact signals for different windows. The window size changes logarithmically
            # i.e., the window size is larger for older samples and shorter for recent samples
            contact_obs[:, i, :] = np.any(contact_hist[:, start:end, :], axis=1)

            # Mean accumulation of the attitude and action signals for different windows. The window size changes logarithmically
            # i.e., the window size is larger for older samples and shorter for recent samples
            att_obs[:, i, :] = np.mean(att_hist[:, start:end, :], axis=1)
            action_obs[:, i, :] = np.mean(action_hist[:, start:end, :], axis=1)

        assert (contact_obs[:, -1, :] == contact_hist[:, -1, :]).all(), f"Last contact in obs {contact_obs[:, -1, :]} should be the same as the last contact in history {contact_hist[:, -1, :]}"
        assert (att_obs[:, -1, :] == att_hist[:, -1, :]).all(), f"Last attitude in obs {att_obs[:, -1, :]} should be the same as the last attitude in history {att_hist[:, -1, :]}"

        return { 'des_dir': self.des_dir,
                'contacts': contact_obs,
                     'att': att_obs,
               'prev_cmds': action_obs}
    
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