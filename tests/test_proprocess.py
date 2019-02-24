import unittest

import gym

from common.preprocess import dqn_preprocess


class TestPreprocess(unittest.TestCase):
    def test_dqn_preprocess(self):
        env = gym.make('BreakoutNoFrameskip-v0')
        env.reset()
        frames = []
        for i in range(5):
            observation, _, _, _ = env.step(env.action_space.sample())
            frames.append(observation)
        result = dqn_preprocess(frames)
        assert result.shape == (128, 128, 4)
