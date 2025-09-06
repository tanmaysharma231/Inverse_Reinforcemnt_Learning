import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
#from __future__ import annotations
from imitation.util import util
import gymnasium as gym
import glob
import os
import time
from imitation.data import rollout
from gym import spaces
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from pettingzoo.sisl import multiwalker_v9
seed = 0
SEED = 0
FAST = True

if FAST:
    N_RL_TRAIN_STEPS = 800_000
else:
    N_RL_TRAIN_STEPS = 2_000_000
def runt():
    env_fn = multiwalker_v9
    env_kwargs = {}
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")
    env = env_fn.env(render_mode='none', **env_kwargs)
    #venv = util.make_vec_env("multiwalker_v9", n_envs=4, rng=np.random.default_rng())
    # max(glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime )

    expert = load_policy("ppo", venv=env, path="mv1.zip")
    #expert=PPO.load("mv1")
    rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=np.random.default_rng(SEED),
    )
    learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0005,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
    seed=SEED,
    )
    reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
    )
    airl_trainer = AIRL(
         demonstrations=rollouts,
        demo_batch_size=2048,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )


    env.reset(seed=SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True
    )
    airl_trainer.train(N_RL_TRAIN_STEPS)
    env.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True
    )
    print(
    "Rewards before training:",
    np.mean(learner_rewards_before_training),
    "+/-",
    np.std(learner_rewards_before_training),
)
    print(
    "Rewards after training:",
    np.mean(learner_rewards_after_training),
    "+/-",
    np.std(learner_rewards_after_training),)


if __name__ == "__main__":
    
    runt()


