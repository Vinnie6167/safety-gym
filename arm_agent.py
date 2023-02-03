#!/usr/bin/env python

import argparse
import gym
import safety_gym  # noqa
import numpy as np  # noqa
from safety_gym.envs.engine import Engine

config = {
    'robot_base': 'xmls/arm.xml',
    'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
    'goal_size': 0.5,  # Radius of the goal area (if using task 'goal')
    'robot_locations': [(0,0)],
    'walls_num': 0,
    'gremlins_num': 5,
    # 'gremlins_num': 10,
    # 'walls_locations': [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
}

def run_random(env_name):
    env = Engine(config)
    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            ep_ret, ep_cost = 0, 0
            obs = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        obs, reward, done, info = env.step(act)
        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        env.render()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Safexp-CarPush2-v0')
    args = parser.parse_args()
    run_random(args.env)
