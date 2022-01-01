from gym.envs.registration import register

register(
    id='DinoGame-v0',
    entry_point='gym_env.envs:DinoEnv',
)
