'''
Solve FrozenLake-v0 with Q-Learning

Hyperparameters:
- alpha
- episilon in episilon-soft policy

Steps:
- initialize all Q(S,A) to 0; initialize hyper parameters
- set pi to be episilon-soft, with episilon = 0.1
- for each step S of episode:
    - take A from argmax(Q(S)) with episilon-soft, observe R, S'
    - Q(S,A) <- Q(S,A) + alpha * (R + lambda * max(Q(S',a)) - Q(S,A))
    - S <- S'; NOTE: we do not update A
    - until S is terminal
- repeat until Q converges
- render final policy with no episilon-soft, this should solve the environment
'''

import random

import gym
import numpy as np

from common.policies import episilon_greedy_policy

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

    for i in range(int(1e5)):
        s = env.reset()
        total_update = 0
        while True:
            a = episilon_greedy_policy(q, s)
            s_prime, reward, done, info = env.step(a)
            if done:
                q[s_prime][:] = 0
            max_q = np.max(q[s_prime])
            q_update = alpha * (reward + discount*max_q - q[s][a])
            total_update += q_update
            q[s][a] += q_update
            s = s_prime
            if done:
                if i % int(1e4) == 0 and debug:
                    print("episode %d, total update %f" % (i, total_update))
                break
    return q

if __name__ == '__main__':
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)

    # initialize
    q = np.zeros((env.observation_space.n, env.action_space.n))
    hyper_parameters = {'alpha':0.1, 'discount':1}
    q = train(env, q, hyper_parameters, debug=True)

    # test the trained policy, final reward should be 1
    observation = env.reset()
    for t in range(100):
        env.render()
        action = episilon_greedy_policy(q, observation, greedy=True)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Final reward {}".format(reward))
            print("Final Q:", q)
            break

'''
Sample output: 
Episode finished after 13 timesteps
Final reward 1.0
Final Q: [[0.48686972 0.         0.2210633  0.14866253]
 [0.35226475 0.21856292 0.36125896 0.59094109]
 [0.50832806 0.39312108 0.34535931 0.37701777]
 [0.20845344 0.03684925 0.05642082 0.07237555]
 [0.58567865 0.53386665 0.43186359 0.5173467 ]
 [0.         0.         0.         0.        ]
 [0.21221457 0.18339856 0.45453052 0.12886953]
 [0.         0.         0.         0.        ]
 [0.58075784 0.61196707 0.50082677 0.6869848 ]
 [0.54662029 0.73901902 0.50072503 0.49434762]
 [0.68664874 0.50387436 0.45450302 0.30514138]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.61036222 0.79938585 0.84821566 0.47519609]
 [0.76289985 0.91838441 0.90746685 0.83933814]
 [0.         0.         0.         0.        ]]
'''