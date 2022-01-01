import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
import time

import gym_env  # import this directory to create environment


def make_env(env_id, seed):
    env = gym.make(env_id)
    env.seed(seed)
    # check_env(_env)
    env = Monitor(env)
    print('successfully create gym environment')
    return env


model_path = '/Users/thakorns/Desktop/Eyp/codebases/rl-dino-gym/training_pipeline/models/2ff55tdz/model.zip'
model = DQN.load(path=model_path, device='cpu')

env = make_env(env_id='DinoGame-v0', seed=0)

for i in range(5):
    obs = env.reset()
    ep_reward = 0
    while True:
        _action = model.predict(obs, deterministic=True)
        action = _action[0].item()
        env.render()
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        time.sleep(0.1)

        if done:
            print(f'trial-{i}: reward={ep_reward}, info={info}\n')
            break
