'''
Solve FrozenLake-v0 with Q-Learning with minimum 6 steps

Hyperparameters:
- alpha
- epsilon in epsilon-soft policy

Steps:
- initialize all Q(S,A) to 0; initialize hyper parameters
- set pi to be epsilon-soft, with epsilon = 0.1
- for each step S of episode:
    - take A from argmax(Q(S)) with epsilon-soft, observe R, S'
    - Q(S,A) <- Q(S,A) + alpha * (R + lambda * max(Q(S',a)) - Q(S,A))
    - S <- S'; NOTE: we do not update A
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

    starting_epsilon = 1 / env.action_space.n # important that we start being very exploratory

    timesteps = 1e4
    episodes = 0

    s = env.reset()
    total_update = 0

    for i in range(int(timesteps)):
        epsilon = epsilon_decay(starting_epsilon, timesteps, i)
        a = epsilon_greedy_policy(q, s, epsilon=epsilon)
        s_prime, reward, done, info = env.step(a)
        if done:
            q[s_prime][:] = 0
        max_q = np.max(q[s_prime])
        q_update = alpha * (reward + discount*max_q - q[s][a])
        total_update += q_update
        q[s][a] += q_update
        s = s_prime
        if done:
            # start a new episode
            episodes += 1
            s = env.reset()
            total_update = 0

    print("training complete, total episodes %d, final update %f " % (episodes, total_update)) # 1315 episodes
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
Final Q:
[[1.         1.         1.         1.        ]
 [1.         0.         1.         1.        ]
 [1.         1.         1.         1.        ]
 [1.         0.         1.         1.        ]
 [1.         1.         0.         1.        ]
 [0.         0.         0.         0.        ]
 [0.         1.         0.         1.        ]
 [0.         0.         0.         0.        ]
 [0.99999737 0.         0.99999261 1.        ]
 [0.99999223 0.99999812 1.         0.        ]
 [1.         1.         0.         1.        ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.99983666 1.         0.99989198]
 [1.         1.         1.         1.        ]
 [0.         0.         0.         0.        ]]
'''