import unittest
import pytest

import gym
from gym import envs

from ..envs import Ad
from ..envs import AdServerEnv


def test_reset():
    env = envs.make('AdServer-v0', num_ads=2, time_series_frequency=10)

    (ads, impressions, clicks) = env.reset('Test')

    # Assert
    assert clicks == 0
    assert impressions == 0
    assert ads == [Ad(0), Ad(1)]


def test_step_no_reward():
    env = envs.make('AdServer-v0', num_ads=2, time_series_frequency=10, reward_policy=lambda x: 0)
    env.reset(scenario_name='Test')

    ((ads, impressions, clicks), reward, done, info) = env.step(0)

    assert clicks == 0
    assert impressions == 1
    assert info == {}
    assert reward == 0
    assert not done
    assert ads == [Ad(0, impressions=1), Ad(1)]


def test_step_with_reward():
    env = envs.make('AdServer-v0', num_ads=2, time_series_frequency=10, reward_policy=lambda x: 1)
    env.reset(scenario_name='Test')

    ((ads, impressions, clicks), reward, done, info) = env.step(1)

    assert clicks == 1
    assert impressions == 1
    assert info == {}
    assert reward == 1
    assert not done
    assert ads == [Ad(0), Ad(1, impressions=1, clicks=1)]


test_reset()
test_step_no_reward()
test_step_with_reward()
