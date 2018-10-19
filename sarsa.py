'''
Solve FrozenLake-v0 with Sarsa with minimum 6 steps

Hyperparameters:
- alpha
- epsilon in epsilon-soft policy

Steps:
- initialize all Q(S,A) to 0; initialize hyper parameters
- set pi to be epsilon-soft, with epsilon = 0.1
- for each step S of episode:
    - take A from argmax(Q(S)) with epsilon-soft, observe R, S'
    - choose A' from argmax(Q(S')) with epsilon-soft
    - Q(S,A) <- Q(S,A) + alpha * (R + lambda * Q(S',A') - Q(S,A))
    - S <- S', A <- A'
    - until S is terminal
- repeat until Q converges
- render final policy with no epsilon-soft, this should solve the environment
'''

import random

import gym
from gym.envs.registration import register
import numpy as np

from common.policies import *

def train(env, q, hyper_parameters, debug=False):
    '''
    train the Sarsa policy
    :param
        env: gym environment
        q: Q(S,A) matrix
    :return: an updated q
    '''
    alpha = hyper_parameters['alpha']
    discount = hyper_parameters['discount']

    timesteps = 1e4
    starting_epsilon = 1 / env.action_space.n # important that we start being very exploratory

    s = env.reset()
    episodes = 0
    total_update = 0

    epsilon = epsilon_decay(starting_epsilon, timesteps, 0)
    a = epsilon_greedy_policy(q, s, epsilon=epsilon)

    for i in range(1, int(timesteps+1)):
        epsilon = epsilon_decay(starting_epsilon, timesteps, i)
        s_prime, reward, done, info = env.step(a)
        if done:
            q[s_prime][:] = 0
        a_prime = epsilon_greedy_policy(q, s_prime, epsilon=epsilon)
        q_update = alpha * (reward + discount*q[s_prime][a_prime] - q[s][a])
        total_update += q_update
        q[s][a] += q_update
        s = s_prime
        a = a_prime
        if done:
            s = env.reset()
            a = epsilon_greedy_policy(q, s, epsilon=epsilon)
            total_update = 0
            episodes += 1

    print("training done; total episodes %i, final update %f" % (episodes, total_update)) # 1406
    return q

if __name__ == '__main__':
    env_name = 'FrozenLake-notSlippery-v0'
    register(id=env_name, entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'is_slippery': False})
    env = gym.make(env_name)

    # initialize
    q = np.zeros((env.observation_space.n, env.action_space.n))
    hyper_parameters = {'alpha':0.1, 'discount':1}
    q = train(env, q, hyper_parameters, debug=True)

    # test the trained policy, final reward should be 1
    observation = env.reset()
    for t in range(100):
        env.render()
        action = epsilon_greedy_policy(q, observation, greedy=True)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Final reward {}".format(reward))
            print("Final Q:")
            print(q)
            break

'''
Sample output: 
Episode finished after 6 timesteps
Final reward 1.0
Final Q: [[0.77311302 0.71094352 0.91454925 0.7827461 ]
 [0.80618836 0.         0.90771566 0.81316732]
 [0.79137627 0.91611362 0.80451343 0.81531942]
 [0.85209155 0.         0.61943572 0.68055591]
 [0.63680207 0.6900417  0.         0.80150173]
 [0.         0.         0.         0.        ]
 [0.         0.91758567 0.         0.89011938]
 [0.         0.         0.         0.        ]
 [0.76771875 0.         0.96907976 0.74969815]
 [0.79044144 0.73901709 0.90690114 0.        ]
 [0.9277173  0.9996899  0.         0.92464385]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.96953419 0.98819386 0.8760814 ]
 [0.96531663 0.99512412 1.         0.94105635]
 [0.         0.         0.         0.        ]]
'''