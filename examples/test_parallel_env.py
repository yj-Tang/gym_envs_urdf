from typing import Callable

import gymnasium as gym
import torch
import numpy as np

from rl_algorithms.ppo_class import PPO, Agent
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.urdf_common.reward import Reward
from urdfenvs.wrappers.customized_flatten_observation import CustomizedFlattenObservation
from urdfenvs.wrappers.customized_clip_action import CustomizedClipAction

def make_env_urdf(robot_name, goal, gamma, render=False):
    def thunk():
        robots = [
            GenericUrdfReacher(urdf=robot_name+".urdf", mode="vel"),
        ]
        env= UrdfEnv(
            dt=0.01,
            robots=robots,
            render=render,
        )
        env.add_goal(goal)

        sensor = FullSensor(['position'], ['position', 'size'], variance=0.0)
        env.add_sensor(sensor, [0])
        env.set_spaces()

        env = CustomizedFlattenObservation(env)
        env = CustomizedClipAction(env)
        return env
    return thunk

robot_name = "pointRobot"
gamma = 0.99
capture_video = True
eval_episodes = 10
cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

envs = gym.vector.SyncVectorEnv([make_env_urdf(robot_name, goal1, gamma, render=False) for i in range(3)])
agent = Agent(envs).to(device)
# agent.load_state_dict(torch.load(model_path, map_location=device))
# agent.eval()

obs, _ = envs.reset()
episodic_returns = []
defaultAction = np.array([[1.335811  , 0.43066153, 1.283855 ], [1.335811  , 0.43066153, 1.283855 ], [1.335811  , 0.43066153, 1.283855 ]])
# defaultAction = np.array([[8.335811  , 8.43066153, 8.283855 ]])

eval_episodes = 800

for i in range(eval_episodes):
    print("i: ", i)

    next_obs, r, terminated, truncated, infos = envs.step(defaultAction)

    obs = next_obs

envs.close()