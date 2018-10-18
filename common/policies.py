import random

import numpy as np

def epsilon_greedy_policy(q, observation, epsilon=0.05, greedy=False):
    '''
    :param
        q: Q(S,A), a state_space*action_space array
        observation: current state S, int
        greedy: when True, ignore epsilon-soft
    :return: A, int
    '''
    if greedy:
        return np.argmax(q[observation])

    most_greedy_action = np.argmax(q[observation])
    actions_count = q[observation].shape[0]
    weights = [epsilon] * actions_count
    weights[most_greedy_action] += 1 - actions_count*epsilon

    return random.choices(list(range(actions_count)), weights=weights, k=1)[0]
