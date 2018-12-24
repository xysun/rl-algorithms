'''
Solve CartPole with REINFORCE

Notes:
    - CartPole instead of MountainCar because it has a shorter episode
    - Policy: softmax over state representation features (`observation`)
    - Critic policy: one hidden layer feed forward network; idea from https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
    - A simple linear state value model over observations does not work
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

alpha_pi = 0.001
alpha_v = 0.005
discount = 1

solved_lookback = 100
solved_threwhold = 295.0


def build_critic():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(24, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=tf.train.AdamOptimizer(learning_rate=alpha_v))
    return model


def train(episodes):
    weights_policy = tf.convert_to_tensor(np.random.rand(state_n, action_n))

    encoder = OneHotEncoder()
    encoder.fit([[e] for e in [i for i in range(action_n)]])

    scores = []

    critic = build_critic()

    def update(observation):
        # so many global variables it hurts :(
        with tf.GradientTape() as t:
            t.watch(weights_policy)
            a = tf.reshape(tf.convert_to_tensor(np.array(observation)), (1, state_n))
            z = tf.matmul(a, weights_policy)
            p = tf.nn.softmax(z)
            action = random.choices(range(env.action_space.n), weights=p.numpy().reshape(action_n, ), k=1)[0]
            action_ohc = np.array(encoder.transform([[action]]).toarray(), dtype='float64')
            target = tf.multiply(tf.reshape(p, (action_n,)), tf.reshape(action_ohc, (action_n,)))
            lp = tf.log(tf.reduce_sum(target))
        return action, lp, t

    for i in range(episodes):
        observation = env.reset()
        rewards = 0

        is_done = False

        I = 1

        while not is_done:

            # first action
            action, lp, t_policy = update(observation)
            state = np.reshape(np.array(observation), (1, state_n))
            state_value = critic.predict(state)[0][0]
            gradient_policy = t_policy.gradient(lp, weights_policy).numpy()
            # next state
            observation, reward, is_done, info = env.step(action=action)
            next_state = np.reshape(np.array(observation), (1, state_n))
            next_state_value = critic.predict(next_state)[0][0]

            if is_done:
                next_state_value = 0
            # update weights
            td_error = reward + discount * next_state_value - state_value
            weights_policy += alpha_pi * td_error * gradient_policy * I
            target = np.reshape(np.array([reward + next_state_value]), (1,1))
            critic.fit(state, target, epochs=1, verbose=0)
            rewards += reward
            I *= discount

        with open('logs/actor_critic_cartpole.csv', 'a') as f:
            f.write("%d,%d\n" % (i, rewards))
        print("episode %d, rewards %d" % (i, rewards))

        scores.append(rewards)
        if len(scores) > solved_lookback:
            # check whether solved or not
            if sum(scores[-solved_lookback:]) / float(solved_lookback) >= solved_threwhold:
                print("Solved at episode ", i)
                break

    return weights_policy


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
