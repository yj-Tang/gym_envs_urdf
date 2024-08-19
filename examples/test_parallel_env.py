from typing import Callable

import gymnasium as gym
import torch
import numpy as np

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

gamma = 0.99
envs = gym.vector.SyncVectorEnv([make_env_urdf("pointRobot", goal1, gamma, render=False) for i in range(3)])

obs, _ = envs.reset()
episodic_returns = []
defaultAction = np.array([[1.  , 0.4, 1. ], 
                          [1.  , 0.4, 1. ], 
                          [1.  , 0.4, 1. ]])

eval_episodes = 200
for i in range(eval_episodes):
    print("i: ", i)

    next_obs, r, terminated, truncated, infos = envs.step(defaultAction)

    obs = next_obs

envs.close()