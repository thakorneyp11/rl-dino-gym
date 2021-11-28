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


def pipeline_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', '-e', help='environment ID', type=str, default='DinoGame-v0')
    parser.add_argument('--seed', '-s', help='reproduce seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(5e5))
    return parser.parse_args()


def init_wandb(api_key, config):
    wandb.login(key=api_key)
    unique_name = f"experiment {pendulum.now(tz='Asia/Bangkok').to_atom_string()}"
    run = wandb.init(
        project="dino-game",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # (optional)
        id=None,  # (optional) unique ID for the run, used for resuming
        name=unique_name,  # (optional) short display name for the run
        tags=["test"],  # (optional) useful for organizing runs together
        notes="",
    )
    return run


def make_env(env_id, seed):
    def _make_env():
        _env = gym.make(env_id)
        _env.seed(seed)
        # check_env(_env)
        _env = Monitor(_env)
        return _env
    env = DummyVecEnv([_make_env])
    print('successfully create gym environment')
    return env


def train_rl_model(env, num_timesteps, config, run):
    # TODO: require to remove the clip-function in stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm.collect_rollouts()
    # TODO: require to change round-function to np.round() in ep_info of stable_baselines3.common.monitor.Monitor.step()
    model = DQN(
        policy=config["policy_type"],
        env=env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        seed=config["seed"],
        device='cpu')

    model.learn(
        total_timesteps=num_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            model_save_freq=1e5,
            verbose=2,
        ),
    )


if __name__ == '__main__':
    args = pipeline_arg_parser()
    working_seed = args.seed
    env_id = args.env_id
    num_timesteps = args.num_timesteps

    wandb_api_key = ""

    config_info = {
        "algorithm": "DQN",
        "policy_type": "CnnPolicy",
        "seed": working_seed
    }

    env = make_env(env_id, working_seed)

    run = init_wandb(wandb_api_key, config_info)

    train_rl_model(env, num_timesteps, config_info, run)

    env.close()
