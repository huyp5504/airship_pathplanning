import os
import gym
from gym import spaces
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from myenv_conv import PathPlanningEnv


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We assume the observation space is a Dict with 'vector' and 'matrix' keys
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        # Extract dimensions for vector and matrix
        self.vector_size = observation_space.spaces['vector'].shape[0]
        self.matrix_shape = observation_space.spaces['matrix'].shape

        # Define the CNN for the matrix
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Calculate the output size of the CNN
        n_flatten = self._get_flattened_size()

        # Define the fully connected layers for the combined features
        self.fc = nn.Sequential(
            nn.Linear(self.vector_size + n_flatten, features_dim),
            nn.ReLU()
        )

    def _get_flattened_size(self):
        # Forward pass to calculate the size of the flattened CNN output
        with th.no_grad():
            dummy_input = th.zeros( 1, *self.matrix_shape)  # ensure number of channels is 1
            n_flatten = self.cnn(dummy_input).shape[1]
        return n_flatten

    def forward(self, observations: dict) -> th.Tensor:
        # Extract vector and matrix from the observation
        vector = observations['vector']
        matrix = observations['matrix'] # Add channel dimension for CNN

        # Process the matrix through the CNN
        matrix_features = self.cnn(matrix)

        # Concatenate the vector and the matrix features
        combined_features = th.cat((vector, matrix_features), dim=1)

        # Process the combined features through fully connected layers
        return self.fc(combined_features)

