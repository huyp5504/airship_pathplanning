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
from model import CustomCombinedExtractor
import matplotlib.pyplot as plt
import netCDF4 as nc
MAX_L = 160

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
        plot[0, i] = obs['vector'][0]  # 记录 x 坐标
        plot[1, i] = obs['vector'][1]  # 记录 y 坐标
        print(i, action, obs['vector'][4:6])
        obs, reward, done, info = env.step(action)
        r += reward
        env.render()
        if done:
            break

    plt.xlim(0, MAX_L)
    plt.ylim(0, MAX_L)
    for i in range(len(plot[0])):
        plot[0, i] = env.end_x - plot[0, i]  # 记录 x 坐标
        plot[1, i] = env.end_y - plot[1, i]  # 记录 y 坐标

    for k in range(10):
        U = env.dataset.variables['u'][k][0][0]
        V = env.dataset.variables['v'][k][0][0]
        for i in range(MAX_L):
            for j in range(MAX_L):
                if i % 10 == 0 and j % 10 == 0:
                    plt.arrow(i, j, U[i, j] / 2 if U[i, j] > 2 else U[i, j], V[i, j] / 2 if V[i, j] > 3 else V[i, j],
                            head_width=0.1, head_length=0.2, fc='white', ec='black')

    plt.plot(plot[0], plot[1], color='red', label='USV 0')  # 绘制路径
    goal = plt.Circle((env.end_x, env.end_y), 1, color='green', fill=False)  # 绘制终点
    start = plt.Circle((env.start_x, env.start_y), 0.1, color='blue', fill=False)
    plt.gca().add_patch(goal)
    plt.gca().add_patch(start)
    plt.axis('equal')  # 保持坐标轴比例
    print(r)
    plt.show()  # 显示图像
    
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
        plot[0, i] = obs['vector'][0]  # 记录 x 坐标
        plot[1, i] = obs['vector'][1]  # 记录 y 坐标
        # print(i, action, obs['vector'][:4])
        obs, reward, done, info = env.step(action)
        r += reward
        env.render()
        if done:
            print(modelpath+"done after "+str(i)+" steps")
            break
    for i in range(len(plot[0])):
        plot[0, i] = env.end_x - plot[0, i]  # 记录 x 坐标
        plot[1, i] = env.end_y - plot[1, i]  # 记录 y 坐标
    return plot

if __name__ == '__main__':
    runwithplot("model\PPO_myenv_cov32_500w",PathPlanningEnv,32)