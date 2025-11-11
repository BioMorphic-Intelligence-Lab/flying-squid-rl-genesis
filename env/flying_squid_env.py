from numpy import dtype
import genesis as gs
import gymnasium as gym
import torch

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
                 corridor_angle_range=torch.deg2rad(torch.tensor([-45, 45])),
                 p_ini=None,
                 yaw_ini=None,
                 debug=False,
                 corridor=True,
                 cylinder_obstacle_density=0.025,
                 box_obstacle_density=0.025):

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
        
        # Set device for tensors
        if str.lower(device) == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')

        # Action Space: [theta, ||v||, omega] aka
        #    - angle and magnitude of desired linear speed in body frame
        #    - desired angular velocity
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        # Observation Space { 'des_dir': [sin(phi), cos(phi)]
        #                    'contacts': [[FR, FL, RL, RR] * observation_length]
        #                         'att': [sin(theta), cos(theta)]
    #                       'prev_cmds': [[theta, ||v||, omega] * observation_length]} 
        observation_space = gym.spaces.Dict({
             'des_dir': gym.spaces.Box(low=-1, high=1, shape=(2,)),
            'contacts': gym.spaces.MultiBinary((observation_length, 4)),
                 'att': gym.spaces.Box(low=-1, high=1, shape=(2,)),
           'prev_cmds': gym.spaces.Box(low=-1, high=1, shape=(observation_length, action_space.shape[0]))
        })

        # Init the Vector Env
        super().__init__(num_envs,
                         observation_space,
                         action_space)

        # Init indeces
        self.envs_idx = torch.arange(self.num_envs)

        # Remember Env Configuration
        self.MAX_STEPS = max_steps
        self.MAX_LIN_VEL = max_lin_vel
        self.MAX_ROT_VEL = max_rot_vel
        self.OBSERVATION_LENTGH = observation_length
        self.HISTORY_DURATION = history_duration
        self.HISTORY_LENGTH = int(torch.ceil(torch.tensor(self.HISTORY_DURATION / self.dt)))
        self.CORRIDOR_WIDTH_RANGE = corridor_width_range
        self.CORRIDOR_ANGLE_RANGE = corridor_angle_range
        self.CORRIDOR_BOX_SIZE = torch.tensor([0.5, 50, 2], device=self.device)
        self.CORRIDOR = corridor
        self.CYLINDER_OBSTACLE_DENSITY = cylinder_obstacle_density
        self.BOX_OBSTACLE_DENSITY = box_obstacle_density
        self.OBSTACTLE_SIZE_RANGE = torch.arange(start=0.3, end=0.8, step=0.2, device=self.device)
        self.FLIGHT_HEIGHT=1.25

        # Init error integral for lin vel control
        self._integral_vel_error = torch.zeros([self.num_envs, 2], device=self.device)
        self._integral_omega_error = torch.zeros([self.num_envs, 1], device=self.device)

        if p_ini is None:
            num_per_side = int(torch.ceil(torch.sqrt(torch.tensor(self.num_envs))))  # Number of rows/cols
            x_idx, y_idx = torch.meshgrid(torch.arange(num_per_side), torch.arange(num_per_side), indexing='ij')
            
            # Select the first num_envs positions
            grid_positions = torch.stack([x_idx.ravel(), y_idx.ravel()], dim=1)[:self.num_envs]
            
            # Center the grid at the origin
            grid_positions = grid_positions * 2 * torch.max(self.CORRIDOR_BOX_SIZE)
            self.P_INI = torch.cat([grid_positions, torch.zeros(self.num_envs).unsqueeze(1)], dim=1)  # Add Z coordinate
        else:
            self.P_INI = p_ini

        if yaw_ini is None:
            self.YAW_INI = torch.tensor([[0] for _ in range(self.num_envs)])
        else:
            self.YAW_INI = yaw_ini
        self.g = g

        # Init step counter to terminate envs
        self.step_counts = torch.zeros(self.num_envs, dtype=torch.int)

        # Init Observation hist
        self.contact_hist = [deque(torch.zeros([self.HISTORY_LENGTH, 4])) for _ in range(self.num_envs)]
        self.att_hist = [deque(torch.zeros([self.HISTORY_LENGTH, 2])) for _ in range(self.num_envs)]
        self.action_hist = [deque(torch.zeros([self.HISTORY_LENGTH, self.action_space.shape[0]])) for _ in range(self.num_envs)]
        self.prev_distance = torch.zeros(self.num_envs)

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
        self.corridor_widths = torch.rand(self.num_envs) * (self.CORRIDOR_WIDTH_RANGE[1] - self.CORRIDOR_WIDTH_RANGE[0]) + self.CORRIDOR_WIDTH_RANGE[0]
        self.corridor_angles = torch.rand(self.num_envs) * (self.CORRIDOR_ANGLE_RANGE[1] - self.CORRIDOR_ANGLE_RANGE[0]) + self.CORRIDOR_ANGLE_RANGE[0]

        # Init obstacle arrays
        self.cylinder_obstacles = [
            [self.scene.add_entity(
                gs.morphs.Cylinder(pos=(0, -5, 0.5 * self.CORRIDOR_BOX_SIZE[2]), euler=(0, 0, 0),
                                    radius=radius, height=self.CORRIDOR_BOX_SIZE[2],
                                    fixed=True)) 
                for _ in range(int(self.CYLINDER_OBSTACLE_DENSITY * self.CORRIDOR_WIDTH_RANGE[1] * self.CORRIDOR_BOX_SIZE[1]))]
            for radius in self.OBSTACTLE_SIZE_RANGE
        ] 
        self.cylinder_obstacle_counters = torch.zeros(len(self.OBSTACTLE_SIZE_RANGE))

        # Init obstacle arrays
        self.box_obstacles = [
            [self.scene.add_entity(
                gs.morphs.Box(pos=(0, -5, 0.5 * self.CORRIDOR_BOX_SIZE[2]),
                              euler=(0, 0, 0),
                              size=(width, width, self.CORRIDOR_BOX_SIZE[2]),
                              fixed=True)
                ) 
                for _ in range(int(self.BOX_OBSTACLE_DENSITY * self.CORRIDOR_WIDTH_RANGE[1] * self.CORRIDOR_BOX_SIZE[1]))]
            for width in self.OBSTACTLE_SIZE_RANGE
        ] 
        self.box_obstacle_counters = torch.zeros(len(self.OBSTACTLE_SIZE_RANGE))

        # Init desired direction
        if self.num_envs == 1:
            theta = torch.zeros([self.num_envs, 1]) #self.corridor_angles + torch.rand(self.num_envs) * (torch.deg2rad(torch.tensor(45.0)) - torch.deg2rad(torch.tensor(-45.0))) + torch.deg2rad(torch.tensor(-45.0))
            self.des_dir = torch.tensor([[torch.sin(theta), torch.cos(theta)]])
        else:
            theta = torch.zeros([self.num_envs]) #self.corridor_angles + torch.rand(self.num_envs) * (torch.deg2rad(torch.tensor(45.0)) - torch.deg2rad(torch.tensor(-45.0))) + torch.deg2rad(torch.tensor(-45.0))
            self.des_dir = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
    

        # Init Genesis Scene
        if self.vis:
            self.drone = self.scene.add_entity(
                gs.morphs.URDF(
                    file="./assets/drone_w_arms.urdf",  # Path to your URDF file
                    fixed=True,
                    merge_fixed_links=False,
                ), visualize_contact=self.debug
            )
        else:
            self.drone = self.scene.add_entity(
                gs.morphs.URDF(
                    file="./assets/drone_w_arms_simple.urdf",  # Path to your URDF file
                    fixed=True,
                    merge_fixed_links=False,
                ), visualize_contact=self.debug
            )

        # Arm parameters
        self.K = 5.0 * torch.eye(10 * 4)
        self.MAX_TENSION = 10.0
        self.RADIUS = 0.2

        # Drone base damping
        self.D = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0]))

        # Finally build the scene
        self.scene.build(n_envs=self.num_envs)

        # Get overall mass
        self.M = self.drone.get_mass()
        init_pos = torch.cat([torch.zeros([self.num_envs, 2]), self.FLIGHT_HEIGHT * torch.ones([self.num_envs, 1]),
                                   self.YAW_INI, torch.zeros([self.num_envs, 10 * 4])], dim=1) 
        self.drone.set_pos(torch.cat([self.P_INI[:, :2], torch.zeros([self.num_envs, 1])], dim=1))
        self.drone.set_dofs_position(init_pos)
        self.drone.set_dofs_velocity(torch.zeros_like(init_pos))

    def _generate_obstacles(self, size_range, env_idx, offset=1.0):

        # Set all obstacle positons back to the original place
        for radius in self.cylinder_obstacles:
            for obstacle in radius:
                pos = torch.tensor([[0, -5, 0.5 * self.CORRIDOR_BOX_SIZE[2]] for _ in range(len(env_idx))])
                obstacle.set_pos(pos, envs_idx=env_idx)

        for size in self.box_obstacles:
            for obstacle in radius:
                pos = torch.tensor([[0, -5, 0.5 * self.CORRIDOR_BOX_SIZE[2]] for _ in range(len(env_idx))])
                obstacle.set_pos(pos, envs_idx=env_idx)

        # Find number of obstacles
        num_cylinders = (self.CYLINDER_OBSTACLE_DENSITY * self.corridor_widths[env_idx] * (self.CORRIDOR_BOX_SIZE[1] - offset)).to(torch.int)
        num_boxes = (self.BOX_OBSTACLE_DENSITY * self.corridor_widths[env_idx] * (self.CORRIDOR_BOX_SIZE[1] - offset)).to(torch.int)
        
        # Init pos vector
        pos = torch.zeros([len(env_idx), 3])

        # Loop over environments
        for idx, n in enumerate(num_cylinders):
            # Loop over number of obstaces
            for i in range(n):
                # Randomly select radius
                radius_idx = torch.randint(size_range[0], size_range[1], (1,)).item()
                # Define rotation matrix z-axis
                cT = torch.cos(self.corridor_angles[env_idx[idx]])
                sT = torch.sin(self.corridor_angles[env_idx[idx]])
                rot = torch.tensor([[cT, -sT, 0],
                                [sT, cT, 0],
                                [0, 0, 1
                ]])
                # Get random location inside corridor within [low, high] for each component
                w = self.corridor_widths[env_idx[idx]]
                r = self.OBSTACTLE_SIZE_RANGE[radius_idx]
                low_vec = torch.tensor([(-0.5 * w + r).item(), float(offset), 0.0])
                high_vec = torch.tensor([(0.5 * w - r).item(), float(self.CORRIDOR_BOX_SIZE[1]), 0.0])
                rand_unit = torch.rand(3)
                rand_vec = low_vec + (high_vec - low_vec) * rand_unit
                pos = self.P_INI[env_idx[idx], :] + rot @ rand_vec
                # Set height
                pos[2] = 0.5 * self.CORRIDOR_BOX_SIZE[2]

                # Set position of obstacle
                self.cylinder_obstacles[radius_idx][i].set_pos(pos.unsqueeze(0), envs_idx=[env_idx[idx]])

                # Init pos vector
        pos = torch.zeros([len(env_idx), 3])

        # Loop over environments
        for idx, n in enumerate(num_boxes):
            # Loop over number of obstaces
            for i in range(n):
                # Randomly select radius
                radius_idx = torch.randint(size_range[0], size_range[1], (1,)).item()
                # Define rotation matrix z-axis
                cT = torch.cos(self.corridor_angles[env_idx[idx]])
                sT = torch.sin(self.corridor_angles[env_idx[idx]])
                rot = torch.tensor([[cT, -sT, 0],
                                [sT, cT, 0],
                                [0, 0, 1
                ]])
                # Get random location inside corridor within [low, high] for each component
                w = self.corridor_widths[env_idx[idx]]
                r = self.OBSTACTLE_SIZE_RANGE[radius_idx]
                low_vec = torch.tensor([(-0.5 * w + r).item(), float(offset), 0.0])
                high_vec = torch.tensor([(0.5 * w - r).item(), float(self.CORRIDOR_BOX_SIZE[1]), 0.0])
                rand_unit = torch.rand(3)
                rand_vec = low_vec + (high_vec - low_vec) * rand_unit
                pos = self.P_INI[env_idx[idx], :] + rot @ rand_vec
                # Set height
                pos[2] = 0.5 * self.CORRIDOR_BOX_SIZE[2]

                # Find quat
                quat = torch.tensor([torch.cos(self.corridor_angles[env_idx[idx]] / 2), # w
                                 0, 0, torch.sin(self.corridor_angles[env_idx[idx]] / 2)] # x y z
                                 )

                # Set position of obstacle
                self.box_obstacles[radius_idx][i].set_pos(pos.unsqueeze(0), envs_idx=[env_idx[idx]])
                self.box_obstacles[radius_idx][i].set_quat(quat.unsqueeze(0), envs_idx=[env_idx[idx]])

    def _generate_corridor(self, width, angle, env_idx):
        # Ensure `angle` is a column vector of shape (len(env_idx), 1) for broadcasting
        cos_a = torch.cos(angle)[:, None]
        sin_a = torch.sin(angle)[:, None]
        zeros_a = torch.zeros_like(angle)[:, None]
        ones_a = torch.ones_like(angle)[:, None]

        # Make `rot` shape (len(env_idx), 3, 3)
        rot = torch.stack([
            torch.cat([cos_a, -sin_a, zeros_a], dim=1),
            torch.cat([sin_a, cos_a, zeros_a], dim=1),
            torch.cat([zeros_a, zeros_a, ones_a], dim=1)
        ], dim=1)  # Shape: (len(env_idx), 3, 3)

        # Broadcastable translation components
        translation = self.P_INI[env_idx, :] * torch.tensor([1, 1, 0])

        # Define vectors with correct shape
        box_offset = torch.tensor([self.CORRIDOR_BOX_SIZE[0], self.CORRIDOR_BOX_SIZE[1], self.CORRIDOR_BOX_SIZE[2]]) - torch.tensor([0.0, 2.0, 0])
        vec1 = torch.stack([-width - box_offset[0], ones_a[:, 0] * box_offset[1], ones_a[:, 0] * box_offset[2]], dim=1)
        vec2 = torch.stack([ width + box_offset[0], ones_a[:, 0] * box_offset[1], ones_a[:, 0] * box_offset[2]], dim=1)

        # Apply rotation for each environment instance
        p1 = translation + torch.einsum('bij,bj->bi', rot, vec1) * 0.5
        p2 = translation + torch.einsum('bij,bj->bi', rot, vec2) * 0.5

        # Ensure shape is (len(env_idx), 3)
        p1 = p1.reshape(len(env_idx), 3)
        p2 = p2.reshape(len(env_idx), 3)

        # Ensure `euler` has shape (len(env_idx), 3)
        euler = torch.stack([
            torch.zeros_like(angle),   # X rotation (zero)
            torch.zeros_like(angle),   # Y rotation (zero)
            torch.rad2deg(angle)       # Z rotation (converted to degrees)
        ], dim=1)  # Shape: (len(env_idx), 3)

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
            self.corridor_widths[dones] = torch.rand(num_resets) * (self.CORRIDOR_WIDTH_RANGE[1] - self.CORRIDOR_WIDTH_RANGE[0]) + self.CORRIDOR_WIDTH_RANGE[0]
            self.corridor_angles[dones] = torch.rand(num_resets) * (self.CORRIDOR_ANGLE_RANGE[1] - self.CORRIDOR_ANGLE_RANGE[0]) + self.CORRIDOR_ANGLE_RANGE[0]

            if self.CORRIDOR:
                self._generate_corridor(width=self.corridor_widths[dones], angle=self.corridor_angles[dones], env_idx=self.envs_idx[dones])

            # Generate obstacles
            self._generate_obstacles(size_range=[0, 2], env_idx=self.envs_idx[dones])

            # Init desired direction
            theta = -self.corridor_angles[dones] + torch.rand(num_resets) * (torch.deg2rad(torch.tensor(45.0)) - torch.deg2rad(torch.tensor(-45.0))) + torch.deg2rad(torch.tensor(-45.0))
            if self.num_envs == 1:
                self.des_dir[dones, :] = torch.tensor([[torch.sin(theta), torch.cos(theta)]])
            else:
                self.des_dir[dones, :] = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)

            # Reset Drone Positions
            init_pos = torch.cat([torch.zeros([num_resets, 2]), self.FLIGHT_HEIGHT * torch.ones([num_resets, 1]),
                                       self.YAW_INI[dones], 0.2 * torch.ones([num_resets, 10 * 4])], dim=1)
            self.drone.set_dofs_position(init_pos, envs_idx=self.envs_idx[dones])
            self.drone.set_dofs_velocity(torch.zeros_like(init_pos), envs_idx=self.envs_idx[dones])
            self.step_counts[dones] = torch.zeros(num_resets, dtype=torch.int32) 

            # Reset obs history
            for i in range(self.num_envs):
                if dones[i]:
                    self.contact_hist[i] = deque(torch.zeros([self.HISTORY_LENGTH, 4]))
                    self.att_hist[i] = deque(torch.zeros([self.HISTORY_LENGTH, 2]))
                    self.action_hist[i] = deque(torch.zeros([self.HISTORY_LENGTH, self.action_space.shape[0]]))
                    # Init last element in action hist with optimal action
                    self.action_hist[i].popleft()
                    self.action_hist[i].append([torch.atan2(self.des_dir[i, 0], self.des_dir[i, 1]), 1.0, 0.0])

        return self._get_observation()

    def reset(self):
        return self.reset_(torch.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.actions = torch.tensor(actions)

    def _lin_vel_ctrl(self, v, v_des, kp=100, ki=25):
        vel_error = v_des[:, :2] - v[:, :2]
        self._integral_vel_error += vel_error * self.dt
        return (kp * vel_error + ki * self._integral_vel_error)

    def _altitude_ctrl(self, h, hdot, h_des, kp=100, kd=25):
        return (
            kp * (h_des - h) - kd*hdot + 
            self.M * self.g * torch.ones_like(h)
        )

    def _rot_vel_ctrl(self, omega, omega_des, kp=50, ki=25):

        omega_error = (omega_des - omega).reshape([self.num_envs, 1])
        self._integral_omega_error += omega_error * self.dt
        torque = kp * omega_error + ki * self._integral_omega_error
        return torque

    def _arm_ctrl(self, alpha):
        tensor_alpha = torch.tensor([alpha],dtype=torch.float32)
        torch.clamp(tensor_alpha, min=-1.0, max=1.0, out=tensor_alpha)
        tension = alpha * self.MAX_TENSION
        return torch.ones([self.num_envs, 10 * 4]) * self.RADIUS * tension

    def step_wait(self):

        # Find current state
        p = self.drone.get_dofs_position()
        v = self.drone.get_dofs_velocity()

        # Increment counter
        self.step_counts += 1

        if torch.max(torch.abs(p)) > 1000:  # Arbitrary large boundary
            print(f"Object out of bounds in environment {torch.nonzero(torch.max(torch.abs(p)))}.")

        # If the simulation broke we stop it
        n_env_broken = torch.isnan(p).any(dim=1).flatten()
        if n_env_broken.any():
            timestep = self.step_counts[torch.nonzero(n_env_broken)].flatten()
            print(f"NaN occured in position in environment {torch.nonzero(n_env_broken)}."
                  + f"At timestep {timestep}={self.dt * timestep} s.")
        
        # Ensure the NaN does not propagate 
        p[n_env_broken, :] = torch.zeros(torch.sum(n_env_broken).item(), p.shape[1])
        v[n_env_broken, :] = torch.zeros(torch.sum(n_env_broken).item(), v.shape[1])
        sim_break_cost = -1e3 * n_env_broken

        # Update Observation hist
        contacts = torch.zeros([self.num_envs, 4])
        contacts[~n_env_broken, :] = 1.0 * (torch.norm(self.drone.get_links_net_contact_force()[~n_env_broken, -4:, :], dim=2) > 0)

        for n in range(self.num_envs):
            self.contact_hist[n].popleft()
            self.contact_hist[n].append(contacts[n, :])
            self.att_hist[n].popleft()
            self.att_hist[n].append(torch.tensor([torch.sin(p[n, 3]), torch.cos(p[n, 3])]))
            self.action_hist[n].popleft()
            self.action_hist[n].append(self.actions[n, :])

        # Translate velocity command action from [-1, 1] to [0, 1]
        self.actions[:, 1] = 0.5 * self.actions[:, 1] + 0.5

        # Clip actions
        self.actions = torch.clamp(
            self.actions * torch.tensor([torch.pi, 1.0, self.MAX_ROT_VEL]),
                     min=torch.tensor([-torch.pi, 0, -self.MAX_ROT_VEL]),
                     max=torch.tensor([torch.pi, 1.0, self.MAX_ROT_VEL])
        ).reshape([self.num_envs, 3])  

        # Extract the commands from the actions
        # The angle is in the body frame so we rotate it accordingly
        angle = torch.atan2(torch.sin(self.actions[:, 0] - p[:, 3]),
                           torch.cos(self.actions[:, 0] - p[:, 3])) 
        v_des = self.actions[:, 1, None] * self.MAX_LIN_VEL * torch.vstack([torch.sin(angle), torch.cos(angle)]).T
        omega_des = self.actions[:, 2]

        # Draw command arrows
        if self.debug:  
            self.scene.clear_debug_objects()
            for n in range(self.num_envs):      
                # Commanded Direction
                self.scene.draw_debug_arrow(self.P_INI[n, :] + p[n, :3], torch.cat([v_des[n, :], torch.zeros(1)]),
                                        radius=0.01, color=(1, 0, 0, 0.5))
                # Commanded Angular Velocity
                self.scene.draw_debug_arrow(self.P_INI[n, :] + p[n, :3],
                                            torch.cat([torch.zeros(2), omega_des[n, None]]),
                                            radius=0.01, color=(1, 1, 0, 0.5))
                # Desired Direction
                self.scene.draw_debug_arrow(self.P_INI[n, :] + p[n, :3], torch.cat([self.des_dir[n, :], torch.zeros(1)]),
                                            radius=0.01, color=(0, 0, 1, 0.5))
        
        # Apply torque within limits
        lin_ctrl = self._lin_vel_ctrl(v=v[:, :2], v_des=v_des)
        h_ctrl = self._altitude_ctrl(h=p[:, 2], hdot=v[:, 2], h_des=self.FLIGHT_HEIGHT).reshape([self.num_envs, 1])
        rot_ctrl = self._rot_vel_ctrl(omega=v[:, 3], omega_des=omega_des).reshape([self.num_envs, 1])
        arm_ctrl = self._arm_ctrl(alpha=0.8)

        # Find stiffness contribution for the arm
        stiffness_contrib = -self.K @ p[:, 4:].T
        # Find damping contribution for the base
        damping_contrib = -self.D @ v[:, :4].T

        # Apply the control, stiffness, and damping forces
        self.drone.control_dofs_force(
            torch.cat([lin_ctrl + damping_contrib[:2, :].T,
                            h_ctrl + damping_contrib[2:3, :].T,
                            rot_ctrl + damping_contrib[3:4, :].T,
                            arm_ctrl + stiffness_contrib.T], dim=1)
        )

        # Episode ends if we reached the max number of steps
        max_steps_reached = self.step_counts >= self.MAX_STEPS
        dones = (max_steps_reached | n_env_broken)

        # Compute reward
        distance = torch.sum(self.des_dir * p[:, :2], dim=1)
        distance_traveled = (distance                                          # Encourage distance traveled along
                              - self.prev_distance)                            # the desired direction
        self.prev_distance = distance
        # Penalize if velocity is too low
        velocity_penalty = -1e2 * (torch.norm(v[:, :2], dim=1) < 0.05).to(torch.float)
        #actuation_cost = (- 1e1 * self.actions[:, 0]**2                        # Penalize deviation from target direction
        #                  - 1e1 * self.actions[:, 1]**2                        # Penalize deviation from max vel
        #                  - 1e1 * self.actions[:, 2]**2)                       # Penalize rotational velocity
        
        #action_hist = torch.stack([
        #    torch.stack([torch.as_tensor(entry, dtype=torch.float32) for entry in self.action_hist[i]], dim=0)
        #    for i in range(self.num_envs)
        #], dim=0)  # [n
        
        
        last_action = torch.stack([list(self.action_hist[i])[-1] for i in range(self.num_envs)], dim=0)
        actuation_variation_cost = - torch.sum((last_action - self.actions)**2 * 1e1 * torch.tensor([1, 1, 1]), dim=1)      # sum to last action
        rewards = (1e3 * distance_traveled #+ actuation_cost
                   + actuation_variation_cost + sim_break_cost + velocity_penalty)
        
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
            if torch.any(torch.isnan(obs[key])) or torch.any(torch.isinf(obs[key])):
                print(f"Warning: NaN or Inf detected in observations[{key}]!")
                print(f"p = {p[torch.isnan(p)]} at index {torch.nonzero(torch.isnan(p))}")
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print("Warning: NaN or Inf detected in rewards!")


        # Convert to numpy before returning because the VecEnvMonitor requires it
        rewards = rewards.cpu().numpy()
        dones = dones.cpu().numpy()
    
        return obs, rewards, dones, infos
    
    def _get_observation(self):     

        # Create edges from where to where to accumulate the history vectors
        edges = torch.linspace(0, self.HISTORY_LENGTH, self.OBSERVATION_LENTGH+1).to(torch.int)

        # Convert lists of deques (per-env time histories) into 3D tensors
        contact_hist = torch.stack([
            torch.stack([torch.as_tensor(entry, dtype=torch.bool) for entry in self.contact_hist[i]], dim=0)
            for i in range(self.num_envs)
        ], dim=0)  # [num_envs, HISTORY_LENGTH, 4]

        att_hist = torch.stack([
            torch.stack([torch.as_tensor(entry, dtype=torch.float32) for entry in self.att_hist[i]], dim=0)
            for i in range(self.num_envs)
        ], dim=0)  # [num_envs, HISTORY_LENGTH, 2]

        action_hist = torch.stack([
            torch.stack([torch.as_tensor(entry, dtype=torch.float32) for entry in self.action_hist[i]], dim=0)
            for i in range(self.num_envs)
        ], dim=0)  # [num_envs, HISTORY_LENGTH, action_dim]

        # Get the most recent attitude as observation
        att_obs = att_hist[:, -1, :]

        # Create empty arrays for contact, angular vels, and action history observations
        contact_obs = torch.zeros(self.num_envs, self.OBSERVATION_LENTGH, 4, dtype=torch.bool)
        action_obs = torch.zeros(self.num_envs, self.OBSERVATION_LENTGH, self.action_space.shape[0])

        for i in range(self.OBSERVATION_LENTGH):
            # Define start and stop indeces
            start = edges[i]
            end = edges[i+1]
            
            # Binary or accumulation of the contact signals for different windows.
            contact_obs[:, i, :] = torch.any(contact_hist[:, start:end, :], dim=1)

            # Mean accumulation of the action signals for different windows.
            action_obs[:, i, :] = torch.mean(action_hist[:, start:end, :], dim=1)

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