import unittest

import numpy as np

from common.policies import *


class TestCommonModule(unittest.TestCase):

    def test_epsilon_greedy_policy(self):

        q = np.array([
            [10,11,12,13] # argmax is action 3
        ])

        epsilon = 0.05

        actual = [0,0,0,0]
        runs = 1e5
        tolerance = runs*epsilon*0.05

        for i in range(int(runs)):
            a = epsilon_greedy_policy(q, observation=0, epsilon=epsilon)
            actual[a] += 1

        for i in range(3):
            self.assertGreaterEqual(actual[i], runs*epsilon-tolerance)
            self.assertLessEqual(actual[i], runs*epsilon+tolerance)

        self.assertGreaterEqual(actual[-1], runs*(1-3*epsilon)-tolerance)
        self.assertLessEqual(actual[-1], runs*(1-3*epsilon)+tolerance)

    def test_greedy_policy(self):

        q = np.array([
            [10,11,12,13]
        ])

        self.assertEqual(epsilon_greedy_policy(q, observation=0, greedy=True), 3)

    def test_epsilon_decay(self):
        self.assertAlmostEqual(epsilon_decay(starting_epsilon=0.1, iterations=100, i=1), 0.1)
        self.assertAlmostEqual(epsilon_decay(starting_epsilon=0.1, iterations=100, i=11), 0.09)
        self.assertAlmostEqual(epsilon_decay(starting_epsilon=0.1, iterations=100, i=100), 0)
