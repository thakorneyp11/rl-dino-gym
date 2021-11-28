import os
import cv2
import numpy as np
import pandas as pd
from gym import Env
from gym import spaces
from gym.utils import seeding

from gym_env.spaces.person import Person
from gym_env.spaces.items import RockItem, BirdItem, EnergyItem

import argparse
import pendulum
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import wandb
from wandb.integration.sb3 import WandbCallback

import gym_env  # import this directory to create environment

import time

def make_env(env_id, seed):
    env = gym.make(env_id)
    env.seed(seed)
    # check_env(_env)
    env = Monitor(env)
    print('successfully create gym environment')
    return env

config = {
        "algorithm": "DQN",
        "policy_type": "CnnPolicy",
        "seed": 0
    }

env = make_env('DinoGame-v0', 0)

model = DQN(
        policy=config["policy_type"],
        env=env,
        verbose=1,
        # tensorboard_log=f"runs/{run.id}",
        seed=config["seed"],
        device='cpu')

for i in range(5):
    env.reset()
    while True:
        action = model.action_space.sample()
        obs, reward, done, info = env.step(action)
        # env.render()
        time.sleep(0.1)
        if done:
            break
