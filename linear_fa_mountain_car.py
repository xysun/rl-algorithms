'''
solve mountain car with linear function approximation

options:
    - sarsa or q learning
    - uniform or asymmetric tilings

https://github.com/dennybritz/reinforcement-learning/issues/76

environment:
    - state: (position: [-1.2, 0.6], velocity ([-0.07, 0.07]))
    - action: [0,1,2]
    - reward: -1.0 per step
    - q(s,a): 8 tilings for each action

algorithm:
- loss = (target - q(s,a)) ^ 2
- where target = R + gamma * q(s', a')
- delta_w = alpha * (target - q(s, a)) * x(s,a)

'''
import random

import gym
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from common.tile_encoding import TileEncoder


def derive_features(observation, action):
    env_features = tile_encoder.encode(observation[0], observation[1])
    empty_features = np.full(env_features.flatten().shape, 0)

    if action == 0:
        return np.concatenate((env_features.flatten(), empty_features, empty_features))
    elif action == 1:
        return np.concatenate((empty_features, env_features.flatten(), empty_features))
    else:
        return np.concatenate((empty_features, empty_features, env_features.flatten()))


env = gym.make('MountainCar-v0').env # to bypass 200 step limit

os = env.observation_space
tile_encoder = TileEncoder(
    lower_x=os.low[0],
    lower_y=os.low[1],
    upper_x=os.high[0],
    upper_y=os.high[1],
    n=8,
    tile_offsets=[]
)

action_encoder = OneHotEncoder(categories='auto')
action_encoder.fit([[0], [1], [2]])

observation = env.reset()

weights = np.full(tile_encoder.d * env.action_space.n, 1.)
alpha = 0.03
gamma = 1


def q(weights, observation, action):
    features = derive_features(observation, action)
    assert features.shape == weights.shape
    return np.dot(weights, features)


def epsilon_greedy(weights, observation, decay = 0, greedy = False):
    q_values = [q(weights, observation, action) for action in [0, 1, 2]]
    epsilon = 0.1 * (1 - decay/10.)

    most_greedy_action: int = np.argmax(q_values)

    if greedy:
        return most_greedy_action

    probs = [epsilon] * 3
    probs[most_greedy_action] += 1 - 3 * epsilon

    return random.choices([0, 1, 2], weights=probs, k=1)[0]


for i in range(0, 110):
    # per episode
    total_reward = 0
    while True:
        action = epsilon_greedy(weights, observation, decay= i % 10)
        q_hat = q(weights, observation, action)
        features = derive_features(observation, action)
        next_observation, reward, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            print("episode %d, total reward %d" % (i, total_reward))
            with open('linear-fa-1.csv', 'a') as f:
                f.write('%d,%d\n'%(i,total_reward))
            observation = env.reset()
            weights += alpha * (reward - q_hat) * features
            break

        next_action = epsilon_greedy(weights, next_observation, decay= i % 10)
        target = reward + gamma * q(weights, next_observation, next_action)
        dw = alpha * (target - q_hat) * features
        weights += dw
        observation = next_observation

observation = env.reset()
for _ in range(1000):
    env.render()
    next_observation, reward, is_done, info = env.step(epsilon_greedy(weights, observation, greedy=True))
    observation = next_observation
    if is_done:
        break
