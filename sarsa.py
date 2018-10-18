'''
Solve FrozenLake-v0 with Sarsa

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

from common.policies import epsilon_greedy_policy

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

    iterations = 1e4
    starting_epsilon = 1 / env.action_space.n
    epsilon_decay_steps = 10.
    epsilon_decay = starting_epsilon / epsilon_decay_steps

    for i in range(int(iterations)):
        s = env.reset()
        epsilon = starting_epsilon - (int(i / (iterations / epsilon_decay_steps)) * epsilon_decay)
        a = epsilon_greedy_policy(q, s, epsilon=epsilon)
        total_update = 0
        while True:
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
                if i % int(iterations/epsilon_decay_steps) == 0 and debug:
                    print("episode %d, total update %f, epsilon %f" % (i, total_update, epsilon))
                break
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
            print("Final Q:", q)
            break

'''
Sample output: 
Episode finished after 59 timesteps
Final reward 1.0
Final Q: 
[[0.14733252 0.12346447 0.11584345 0.11158805]
 [0.0664431  0.08236323 0.07721741 0.10849282]
 [0.12945809 0.1057834  0.08920052 0.09945042]
 [0.0876511  0.08314306 0.06264262 0.10165838]
 [0.15286886 0.08983397 0.09984534 0.08871989]
 [0.         0.         0.         0.        ]
 [0.0954803  0.09900505 0.09994057 0.02226198]
 [0.         0.         0.         0.        ]
 [0.06946377 0.07777983 0.03553089 0.19449307]
 [0.22746631 0.3727177  0.18622745 0.21365838]
 [0.43272015 0.30182448 0.14616152 0.15673924]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.2728878  0.39933574 0.49003632 0.30993405]
 [0.56562515 0.65064389 0.69843132 0.59741935]
 [0.         0.         0.         0.        ]]
'''