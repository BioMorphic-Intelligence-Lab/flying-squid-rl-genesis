import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DualCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_output_dim=5, des_dir_dim=2):
        super().__init__(observation_space, features_dim=1)  # We'll set features_dim later

        # Assume observation_space is a dict with:
        #   "des_dir": desired direction -> MLP
        #   "contacts": history of contact data -> CNN1
        #   "ang_vel": history of angular velocities -> CNN2
        #   "prev_cmds": history of previous commands -> CNN3

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.cnn3 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Run dummy forward to get flattened output size
        with torch.no_grad():
            cnn1_dummy_input = torch.zeros(*observation_space['contacts'].shape)
            cnn2_dummy_input = torch.zeros(*observation_space['ang_vel'].shape)
            cnn3_dummy_input = torch.zeros(*observation_space['prev_cmds'].shape)
            cnn1_out_dim = self.cnn1(cnn1_dummy_input).shape[1]
            cnn2_out_dim = self.cnn2(cnn2_dummy_input).shape[1]
            cnn3_out_dim = self.cnn3(cnn3_dummy_input).shape[1]

        # Total input dim to MLP
        self.total_input_dim = (cnn1_out_dim + cnn2_out_dim
                                + cnn3_out_dim + des_dir_dim)

        # Simple linear MLP with one activation at the end
        self.mlp = nn.Sequential(
            nn.Linear(self.total_input_dim, 32),
            nn.Tanh()
        )

        self._features_dim = 32

    def forward(self, observations):
        x1 = self.cnn1(observations["image1"])
        x2 = self.cnn2(observations["image2"])
        x3 = observations["small_input"]
        x = torch.cat([x1, x2, x3], dim=1)
        return self.mlp(x)
