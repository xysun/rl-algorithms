'''
solve mountain car with linear function approximation

command line options:
    - sarsa or q learning
    - uniform or asymmetric tilings
    - whether use epsilon-greedy or not
compare the performance of all these options (#iterations to solve, variance), and write a blog
and report back to this issue: https://github.com/dennybritz/reinforcement-learning/issues/76

environment:
    - state: (position: [-1.2, 0.6], velocity ([-0.07, 0.07]))
    - action: [0,1,2]
    - reward: -1.0 per step
    - q(s,a): 8 tilings + 1 action, dimension = 9
algorithm:
- loss = (target - q(s,a)) ^ 2
- where target = R + gamma * q(s', a')
- delta_w = alpha * (target - q(s, a)) * x(s,a)

experience replay?

fix weights -> collect a batch of experience -> train SGD -> update weights -> repeat
solution: call clf.predict on 3 actions, pick max
'''
import random

import gym
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from common.tile_encoding import TileEncoder


def derive_features(observation, action):
    env_features = tile_encoder.encode(observation[0], observation[1])
    action_features = action_encoder.transform([[action]]).toarray()[0]

    return np.concatenate((env_features.flatten(), action_features))


env = gym.make('MountainCar-v0')

os = env.observation_space
tile_encoder = TileEncoder(
    lower_x=os.low[0],
    lower_y=os.low[1],
    upper_x=os.high[0],
    upper_y=os.high[1],
    n=4,
    tile_offsets=[]
)

action_encoder = OneHotEncoder(categories='auto')
action_encoder.fit([[0], [1], [2]])

observation = env.reset()

weights = np.full(tile_encoder.d + env.action_space.n, 0.01)
alpha = 0.1
gamma = 1


def q(weights, observation, action):
    features = derive_features(observation, action)
    assert features.shape == weights.shape
    return np.dot(weights, features)


def epsilon_greedy(weights, observation):
    q_values = [q(weights, observation, action) for action in [0, 1, 2]]
    epsilon = 0.2
    most_greedy_action: int = np.argmax(q_values)
    probs = [epsilon] * 3
    probs[most_greedy_action] += 1 - 3 * epsilon

    return random.choices([0, 1, 2], weights=probs, k=1)[0]


for i in range(0, 100):
    # per episode
    total_reward = 0
    while True:
        action = epsilon_greedy(weights, observation)
        q_hat = q(weights, observation, action)
        features = derive_features(observation, action)
        next_observation, reward, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            print("episode %d, total reward %d" % (i, total_reward))
            observation = env.reset()
            weights += alpha * (reward - q_hat) * features
            print(','.join([str(e) for e in weights]))
            break

        next_action = epsilon_greedy(weights, next_observation)
        target = reward + gamma * q(weights, next_observation, next_action)
        dw = alpha * (target - q_hat) * features
        weights += dw
        observation = next_observation
