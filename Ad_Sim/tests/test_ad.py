import unittest
import pytest
from ..envs import adserver

import gym
from gym import envs

from ..envs import Ad


def test_init():
    ad = Ad(3, 2, 1)
    assert ad.id == '3'
    assert ad.impressions == 2
    assert ad.clicks == 1


def test_str():
    assert str(Ad(1, 100, 25)) == 'Ad: 1, CTR: 0.2500'


def test_repr():
    assert repr(Ad(1, 100, 25)) == '(25/100)'


def test_ctr(impressions, clicks, expected):
    assert Ad(1, impressions, clicks).ctr() == expected

test_init()
test_str()
test_repr()
test_ctr(100, 25, 0.25)