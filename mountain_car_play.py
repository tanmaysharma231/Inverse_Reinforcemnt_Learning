import gym
import pygame
from teleop import play
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from pettingzoo.sisl import multiwalker_v9

mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
env = multiwalker_v9.parallel_env(render_mode='rgb_array')
#env = gym.make("multiwalker_v9",render_mode='single_rgb_array') 
demos = play(env, keys_to_action=mapping)