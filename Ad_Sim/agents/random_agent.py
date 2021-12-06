import argparse
import sys
import time
import gym
from gym import wrappers, logger
from ..envs import adserver


class RandomAgent(object):

    def __init__(self, action_space):
        self.name = "Random Agent"
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)
    env.seed(args.seed)

    agent = RandomAgent(env.action_space)

    # Simulation loop
    reward = 0
    done = False
    observation = env.reset(agent.name)
    for i in range(args.impressions):
        action = agent.act(observation, reward, done)
        observation, reward, done, _ = env.step(action)

        observedImpressions = observation[1]
        if observedImpressions % time_series_frequency == 0: 
            env.render()
        
        if done:
            break
    
    env.render(freeze=True, output_file=args.output_file)
    
    env.close()
