'''
solve mountain car with linear function approximation with sklearn's SGDRegressor

options:
    - sarsa or q learning
    - uniform or asymmetric tilings

https://github.com/dennybritz/reinforcement-learning/issues/76

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

estimators = [
    SGDRegressor(learning_rate='constant', eta0=0.03),
    SGDRegressor(learning_rate='constant', eta0=0.03),
    SGDRegressor(learning_rate='constant', eta0=0.03)
]

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

observation = env.reset()
for model in estimators:
    model.partial_fit(
        [tile_encoder.encode(observation[0], observation[1]).flatten()],
        [0]
    )

alpha = 0.03
gamma = 1


def q(observation, action):
    model = estimators[action]
    return model.predict([
        tile_encoder.encode(observation[0], observation[1]).flatten()
    ])[0]


def epsilon_greedy(observation, decay = 0, greedy = False):
    q_values = [q(observation, action) for action in [0, 1, 2]]
    epsilon = 0.1 * (1 - decay/10.)

    most_greedy_action: int = np.argmax(q_values)

    if greedy:
        return most_greedy_action

    probs = [epsilon] * 3
    probs[most_greedy_action] += 1 - 3 * epsilon

    return random.choices([0, 1, 2], weights=probs, k=1)[0]


action = epsilon_greedy(observation, decay= 0)

f = open('linear-fa-1.csv', 'w')

for i in range(0, 100):
    # per episode
    total_reward = 0
    # k = 0
    while True:
        # k += 1
        # if k % 100 == 0:
        #     print("step %d for episode %d" % (k, i))
        q_hat = q(observation, action)
        next_observation, reward, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            print("episode %d, total reward %d" % (i, total_reward))
            f.write('%d,%d\n'%(i,total_reward))
            observation = env.reset()
            break

        next_action = epsilon_greedy(next_observation, decay= i % 10) # SARSA
        target = reward + gamma * q(next_observation, next_action) # SARSA
        # target = reward + gamma * np.max(np.array([q(next_observation, a) for a in [0,1,2]])) # Q-learning
        estimators[action].partial_fit([
            tile_encoder.encode(observation[0], observation[1]).flatten()
        ],[target])
        observation = next_observation
        action = next_action #SARSA
        # action = epsilon_greedy(next_observation, decay= i % 10) # Q-learning

f.close()

observation = env.reset()
for _ in range(1000):
    env.render()
    next_observation, reward, is_done, info = env.step(epsilon_greedy(observation, greedy=True))
    observation = next_observation
    if is_done:
        break
