import unittest

import numpy as np

from sarsa import episilon_greedy_policy


class TestSarsaModule(unittest.TestCase):

    def test_episilon_greedy_policy(self):

        q = np.array([
            [10,11,12,13] # argmax is action 3
        ])

        episilon = 0.05

        actual = [0,0,0,0]
        runs = 1e5
        tolerance = runs*episilon*0.05

        for i in range(int(runs)):
            a = episilon_greedy_policy(q, observation=0, episilon=episilon)
            actual[a] += 1

        for i in range(3):
            self.assertGreaterEqual(actual[i], runs*episilon-tolerance)
            self.assertLessEqual(actual[i], runs*episilon+tolerance)

        self.assertGreaterEqual(actual[-1], runs*(1-3*episilon)-tolerance)
        self.assertLessEqual(actual[-1], runs*(1-3*episilon)+tolerance)

    def test_greedy_policy(self):

        q = np.array([
            [10,11,12,13]
        ])

        self.assertEqual(episilon_greedy_policy(q, observation=0, greedy=True), 3)
