import gym
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from pettingzoo.sisl import multiwalker_v9

env = gym.make('multiwalker_v9')

env.reset()

done = False
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    
env.close()