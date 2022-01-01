from collections import deque
import gym
from gym import spaces
import numpy as np

class ConcatObs(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.concat_obs = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((k,) + shp), dtype=env.observation_space.dtype)
        print('Wrapping the env in ConcatObs with 6-lookback observation_space')

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.concat_obs.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.concat_obs.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.array(self.concat_obs)
