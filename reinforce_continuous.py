'''
Solve MountainCar continuous with REINFORCE
Does not seem to converge as of episode 500 :(
'''

import gym
import numpy as np

from common.tile_encoding import TileEncoder

env = gym.make('MountainCarContinuous-v0')  # .env
state_space = env.observation_space

# same featurizer for both mean and variance
tile_encoder = TileEncoder(
    lower_x=state_space.low[0],
    lower_y=state_space.low[1],
    upper_x=state_space.high[0],
    upper_y=state_space.high[1],
    n=12,
    tile_offsets=[]
)

# linear for mean
alpha_miu = 0.001
alpha_sigma = 0.0001


def predict_miu(observation, weights):
    features = tile_encoder.encode(observation[0], observation[1]).flatten()
    return np.dot(weights, features)


def predict_sigma(observation, weights):
    features = tile_encoder.encode(observation[0], observation[1]).flatten()
    e = np.exp(-1 * np.dot(weights, features))
    # return 1 / (1+e)
    return np.exp(np.dot(weights, features))


def clip(action):
    if action > 1.0:
        return 1.0
    if action < -1.0:
        return -1.0
    return action


def train(episodes):
    weights_miu = np.random.rand(tile_encoder.d)
    weights_sigma = np.random.rand(tile_encoder.d)

    gradients_miu = []
    gradients_sigma = []

    for i in range(episodes):
        observation = env.reset()
        rewards = []

        # debug on
        sigmas = []
        # debug off

        is_done = False
        while not is_done:
            miu = predict_miu(observation, weights_miu)
            state_features = tile_encoder.encode(observation[0], observation[1]).flatten()
            sigma = predict_sigma(observation, weights_sigma)  # + 1e-5
            sigmas.append(sigma)
            action = np.random.normal(miu, sigma)
            # gradients
            gradients_miu.append(1 / (sigma ** 2) * (action - miu) * state_features)
            gradients_sigma.append(((action - miu) ** 2 / (sigma ** 2) - 1) * state_features)
            # gradients_sigma.append((1-sigma) * ((action - miu)**2 / (sigma**2) - 1))
            # step
            observation, reward, is_done, info = env.step([clip(action)])  # clip so that reward does not explode
            rewards.append(reward)

        for j in range(len(rewards)):
            g = sum(rewards[j:])
            gradient_miu = gradients_miu[j]
            gradient_sigma = gradients_sigma[j]

            weights_miu += alpha_miu * g * gradient_miu
            weights_sigma += alpha_sigma * g * gradient_sigma

        # with open('logs/reinforce_continuous_mountaincar.csv', 'a') as f:
        #     f.write("%d,%d" % (i, len(rewards)))
        # print("sigma stats:", max(sigmas), sum(sigmas) / len(sigmas))
        print("episode %d, rewards %d, %.2f" % (i, len(rewards), sum(rewards)))

    return weights_miu, weights_sigma


if __name__ == '__main__':
    episodes = 5000
    weights_miu, weights_sigma = train(episodes)

    # demonstrate
    observation = env.reset()
    for _ in range(1000):
        env.render()
        miu = predict_miu(observation, weights_miu)
        sigma = predict_sigma(observation, weights_sigma)
        action = np.random.normal(miu, sigma)
        observation, reward, is_done, info = env.step([clip(action)])
        if is_done:
            break
