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

tf.enable_eager_execution()

env = gym.make('CartPole-v1')


def train(episodes):
    weights = tf.convert_to_tensor(np.random.rand(4, 2))

    for i in range(episodes):
        observation = env.reset()
        rewards = []
        pis = []
        observations = []
        gradients = []

        # first action
        observations.append(observation)

        with tf.GradientTape() as t:
            t.watch(weights)
            a = tf.reshape(tf.convert_to_tensor(np.array(observation)), (1,4))
            z = tf.matmul(a,weights)
            p = tf.nn.softmax(z)
            action = random.choices([0, 1], weights=p.numpy().reshape(2, ), k=1)[0]
            action_ohc = tf.constant([1.0,0.0], dtype='float64') if action == 0 else tf.constant([0.0,1.0], dtype='float64')
            target = tf.multiply(tf.reshape(p, (2,)), tf.reshape(action_ohc, (2,)))
            lp = tf.log(tf.reduce_sum(target))

        gradient = t.gradient(lp, weights).numpy()
        gradients.append(gradient)
        observation, reward, is_done, info = env.step(action=action)
        rewards.append(reward)

        while not is_done:
            observations.append(observation)

            with tf.GradientTape() as t:
                t.watch(weights)
                a = tf.reshape(tf.convert_to_tensor(np.array(observation)), (1, 4))
                z = tf.matmul(a, weights)
                p = tf.nn.softmax(z)
                action = random.choices([0, 1], weights=p.numpy().reshape(2, ), k=1)[0]
                action_ohc = tf.constant([1.0, 0.0], dtype='float64') if action == 0 else tf.constant([0.0, 1.0],
                                                                                                      dtype='float64')
                target = tf.multiply(tf.reshape(p, (2,)), tf.reshape(action_ohc, (2,)))
                lp = tf.log(tf.reduce_sum(target))

            gradient = t.gradient(lp, weights).numpy()
            gradients.append(gradient)
            observation, reward, is_done, info = env.step(action=action)
            rewards.append(reward)

        alpha = 0.001  # learning rate
        discount = 1

        for j in range(0, len(rewards)):
            g = sum(rewards[j:])
            gradient = gradients[j]
            weights += alpha * g * pow(discount, j) * gradient

        # if i % 1000 == 0:
        print("episode %d, rewards %d" % (i, sum(rewards)))


if __name__ == '__main__':
    episodes = 1000000
    train(episodes)
