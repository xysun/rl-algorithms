'''
Solve CartPole with REINFORCE

Notes:
    - CartPole instead of MountainCar because it has a shorter episode
    - Policy: softmax over state representation features (`observation`)
'''

import random

import gym
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

tf.enable_eager_execution()

env = gym.make('CartPole-v1')
state_n = env.observation_space.shape[0]
action_n = env.action_space.n

alpha = 0.0005  # learning rate
discount = 1

solved_lookback = 100
solved_threwhold = 295.0


def train(episodes):
    weights = tf.convert_to_tensor(np.random.rand(state_n, action_n))
    encoder = OneHotEncoder()
    encoder.fit([[e] for e in [i for i in range(action_n)]])

    scores = []

    def update(observation):
        # so many global variables it hurts :(
        with tf.GradientTape() as t:
            t.watch(weights)
            a = tf.reshape(tf.convert_to_tensor(np.array(observation)), (1, state_n))
            z = tf.matmul(a, weights)
            p = tf.nn.softmax(z)
            action = random.choices(range(env.action_space.n), weights=p.numpy().reshape(action_n, ), k=1)[0]
            action_ohc = np.array(encoder.transform([[action]]).toarray(), dtype='float64')
            target = tf.multiply(tf.reshape(p, (action_n,)), tf.reshape(action_ohc, (action_n,)))
            lp = tf.log(tf.reduce_sum(target))
        return action, lp, t

    for i in range(episodes):
        observation = env.reset()
        rewards = []
        observations = []
        gradients = []

        # first action
        observations.append(observation)
        action, lp, t = update(observation)
        gradient = t.gradient(lp, weights).numpy()
        gradients.append(gradient)
        observation, reward, is_done, info = env.step(action=action)
        rewards.append(reward)

        while not is_done:
            observations.append(observation)
            action, lp, t = update(observation)
            gradient = t.gradient(lp, weights).numpy()
            gradients.append(gradient)
            observation, reward, is_done, info = env.step(action=action)
            rewards.append(reward)

        for j in range(0, len(rewards)):
            g = sum(rewards[j:])
            gradient = gradients[j]
            weights += alpha * g * pow(discount, j) * gradient

        with open('logs/reinforce_cartpole.csv', 'a') as f:
            f.write("%d,%d\n" % (i, sum(rewards)))
        print("episode %d, rewards %d" % (i, sum(rewards)))

        scores.append(sum(rewards))
        if len(scores) > solved_lookback:
            # check whether solved or not
            if sum(scores[-solved_lookback:]) / float(solved_lookback) >= solved_threwhold:
                print("Solved at episode ", i)
                break

    return weights


if __name__ == '__main__':
    episodes = 5000
    weights = train(episodes)

    # demonstrate
    observation = env.reset()
    for _ in range(1000):
        env.render()
        a = tf.reshape(tf.convert_to_tensor(np.array(observation)), (1, state_n))
        z = tf.matmul(a, weights)
        p = tf.nn.softmax(z)
        action = random.choices(range(env.action_space.n), weights=p.numpy().reshape(action_n, ), k=1)[0]
        next_observation, reward, is_done, info = env.step(action)
        observation = next_observation
        if is_done:
            break
