'''
solve mountain car with linear function approximation with sklearn's SGDRegressor and tile encoding

options:
    - sarsa or q learning
    - uniform or asymmetric tilings

environment:
    - state: (position: [-1.2, 0.6], velocity ([-0.07, 0.07]))
    - action: [0,1,2]
    - reward: -1.0 per step
    - q(s,a): 8 tilings for each action
'''

import random

import gym
import numpy as np
from sklearn.linear_model import SGDRegressor

from common.tile_encoding import TileEncoder


class LinearFA(object):
    def __init__(self, env, hyper_parameters):
        # one estimator per action
        alpha = hyper_parameters['alpha']
        self.estimators = [
            SGDRegressor(learning_rate='constant', eta0=alpha),
            SGDRegressor(learning_rate='constant', eta0=alpha),
            SGDRegressor(learning_rate='constant', eta0=alpha)
        ]

        os = env.observation_space
        self.tile_encoder = TileEncoder(
            lower_x=os.low[0],
            lower_y=os.low[1],
            upper_x=os.high[0],
            upper_y=os.high[1],
            n=8,
            tile_offsets=[] # you can add more tiles, see tests.test_tile_encoding.test_multiple_offsets for example
        )

        # pre-fit
        self.first_observation = env.reset()
        for model in self.estimators:
            model.partial_fit(
                [self.tile_encoder.encode(self.first_observation[0], self.first_observation[1]).flatten()],
                [0]
            )

        self.hyper_parameters = hyper_parameters

    def q(self, observation, action):
        model = self.estimators[action]
        return model.predict([
            self.tile_encoder.encode(observation[0], observation[1]).flatten()
        ])[0]

    def get_epsilon_greedy_action(self, observation, decay=0, greedy=False):
        q_values = [self.q(observation, action) for action in [0, 1, 2]]
        epsilon = 0.1 * (1 - decay / 10.)

        most_greedy_action: int = np.argmax(q_values)

        if greedy:
            return most_greedy_action

        chances = [epsilon] * 3
        chances[most_greedy_action] += 1 - 3 * epsilon

        return random.choices([0, 1, 2], weights=chances, k=1)[0]

    def train(self, record_output=False):

        gamma = self.hyper_parameters['gamma']
        mode = self.hyper_parameters['mode']

        assert mode == 'sarsa' or mode == 'q-learning'

        action = self.get_epsilon_greedy_action(self.first_observation, decay=0)
        observation = self.first_observation

        rewards_log = []

        for i in range(0, 100):
            # per episode
            total_reward = 0
            while True:
                next_observation, reward, is_done, info = env.step(action)
                total_reward += reward
                if is_done:
                    print("episode %d, total reward %d" % (i, total_reward))
                    rewards_log.append("%d,%d\n" % (i, total_reward))
                    observation = env.reset()
                    break

                if mode == 'sarsa':
                    next_action = self.get_epsilon_greedy_action(next_observation, decay=i % 10)
                    target = reward + gamma * self.q(next_observation, next_action)
                else:
                    target = reward + gamma * np.max(np.array([self.q(next_observation, a) for a in [0, 1, 2]]))

                self.estimators[action].partial_fit([
                    self.tile_encoder.encode(observation[0], observation[1]).flatten()
                ], [target])

                observation = next_observation
                if mode == 'sarsa':
                    action = next_action
                else:
                    action = self.get_epsilon_greedy_action(next_observation, decay=i % 10)

        if record_output:
            with open('linear-fa-1.csv', 'w') as f:
                for r in rewards_log:
                    f.write(r)


if __name__ == '__main__':
    env = gym.make('MountainCar-v0').env  # to bypass 200 step limit
    hyper_parameters = {
        'alpha': 0.03,  # learning rate
        'gamma': 1,  # discount
        'mode': 'sarsa' # sarsa or q-learning
    }

    trainer = LinearFA(env, hyper_parameters)
    trainer.train()

    # demo
    observation = env.reset()
    for _ in range(1000):
        env.render()
        next_observation, reward, is_done, info = env.step(trainer.get_epsilon_greedy_action(observation, greedy=True))
        observation = next_observation
        if is_done:
            break
