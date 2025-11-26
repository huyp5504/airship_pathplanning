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
from myenv_conv_energy import PathPlanningEnv_energy
import matplotlib.pyplot as plt
import netCDF4 as nc
MAX_L = 160

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

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=n_flatten, hidden_size=256, num_layers=1, batch_first=True)

        # Define the fully connected layers for the combined features
        self.fc = nn.Sequential(
            nn.Linear(self.vector_size + 256, features_dim),
            nn.ReLU()
        )

    def _get_flattened_size(self):
        # Forward pass to calculate the size of the flattened CNN output
        with th.no_grad():
            dummy_input = th.zeros(1, *self.matrix_shape)  # ensure number of channels is 1
            n_flatten = self.cnn(dummy_input).shape[1]
        return n_flatten

    def forward(self, observations: dict) -> th.Tensor:
        # Extract vector and matrix from the observation
        vector = observations['vector']
        matrix = observations['matrix']  # Add channel dimension for CNN

        # Process the matrix through the CNN
        matrix_features = self.cnn(matrix)

        # Reshape matrix_features to (batch_size, seq_len, input_size) for LSTM
        batch_size = matrix_features.size(0)
        seq_len = 1  # Assuming we are processing one step at a time
        matrix_features = matrix_features.view(batch_size, seq_len, -1)

        # Process the matrix features through the LSTM
        lstm_out, _ = self.lstm(matrix_features)
        lstm_out = lstm_out.squeeze(1)  # Remove the sequence length dimension

        # Concatenate the vector and the LSTM output
        combined_features = th.cat((vector, lstm_out), dim=1)

        # Process the combined features through fully connected layers
        return self.fc(combined_features)



# Define the policy with the custom feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[256, 256]
)


def train_and_save(pathplanning_env,env_conv_size, model_name, n_epochs ,policy_kwargs=policy_kwargs):
    env = pathplanning_env(env_conv_size)
    model = PPO("MultiInputPolicy", 
            env, 
            policy_kwargs=policy_kwargs, 
            learning_rate=2e-5,
            verbose=1, 
            tensorboard_log="./tensorboard/training_log")
    model.learn(total_timesteps=n_epochs)
    model.save(model_name)


def runwithplot(modelpath,pathplanning_env,env_conv_size,config=[0,0,0,0]):
    startx,starty,endx,endy=config
    env = pathplanning_env(env_conv_size)
    model = PPO.load(modelpath, env=env)
    obs = env.reset(startx,starty,endx,endy)
    ac = []
    r = 0
    plot = np.zeros((2, env.max_steps))
    for i in range(env.max_steps):
        action, _state = model.predict(obs, deterministic=True)
        plot[0, i] = obs['vector'][0]  # record x coordinate
        plot[1, i] = obs['vector'][1]  # record y coordinate
        print(i, action, obs['vector'][4:6])
        obs, reward, done, info = env.step(action)
        r += reward
        env.render()
        if done:
            break

    plt.xlim(0, MAX_L)
    plt.ylim(0, MAX_L)
    for i in range(len(plot[0])):
        plot[0, i] = env.end_x - plot[0, i]  # record x coordinate
        plot[1, i] = env.end_y - plot[1, i]  # record y coordinate

    for k in range(10):
        U = env.dataset.variables['u'][k][0][0]
        V = env.dataset.variables['v'][k][0][0]
        for i in range(MAX_L):
            for j in range(MAX_L):
                if i % 10 == 0 and j % 10 == 0:
                    plt.arrow(i, j, U[i, j] / 2 if U[i, j] > 2 else U[i, j], V[i, j] / 2 if V[i, j] > 3 else V[i, j],
                            head_width=0.1, head_length=0.2, fc='white', ec='black')

    plt.plot(plot[0], plot[1], color='red', label='USV 0')  # plot path
    goal = plt.Circle((env.end_x, env.end_y), 1, color='green', fill=False)  # plot goal
    start = plt.Circle((env.start_x, env.start_y), 0.1, color='blue', fill=False)
    plt.gca().add_patch(goal)
    plt.gca().add_patch(start)
    plt.axis('equal')  # keep axis aspect ratio
    print(r)
    plt.show()  # show image
    
def run(modelpath,pathplanning_env,env_conv_size,config=[0,0,0,0]):
    startx,starty,endx,endy=config
    env = pathplanning_env(env_conv_size)
    model = PPO.load(modelpath, env=env)
    obs = env.reset(startx,starty,endx,endy)
    ac = []
    r = 0
    plot = np.zeros((2, env.max_steps))
    for i in range(env.max_steps):
        action, _state = model.predict(obs, deterministic=True)
        plot[0, i] = obs['vector'][0]  # record x coordinate
        plot[1, i] = obs['vector'][1]  # record y coordinate
        # print(i, action, obs['vector'][:4])
        obs, reward, done, info = env.step(action)
        r += reward
        env.render()
        if done:
            print(modelpath+"done after "+str(i)+" steps")
            break
    for i in range(len(plot[0])):
        plot[0, i] = env.end_x - plot[0, i]  # record x coordinate
        plot[1, i] = env.end_y - plot[1, i]  # record y coordinate
    return plot

if __name__ == '__main__':
    path = os.path.join( "model","PPO_energyenv_conv32_500w")
    runwithplot(path,PathPlanningEnv_energy,32)
