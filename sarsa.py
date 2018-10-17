'''
Solve FrozenLake-v0 with Sarsa

Hyperparameters:
- alpha
- episilon in episilon-soft policy
- episilon in Q convergence

Questions:
- does it have to be episilon-greedy?

Steps:
- initialize all Q(S,A) to 0; we have 16 states and 4 actions; set alpha to 0.1
- set pi to be episilon-soft, with episilon = 0.1
- for each step S of episode:
    - take A from argmax(Q(S)) with episilon-soft, observe R, S'
    - choose A' from argmax(Q(S')) with episilon-soft
    - Q(S,A) <- Q(S,A) + alpha * (R + lambda * Q(S',A') - Q(S,A))
    - until S is terminal
- repeat until Q converges
- render final policy with no episilon-soft
'''

import gym

env_name = 'FrozenLake-v0'
env = gym.make(env_name)

print("action space:", env.action_space) # LEFT,DOWN,RIGHT,UP
print("state space:", env.observation_space)

for i_episode in range(1):
    observation = env.reset()

    for t in range(100):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(action, reward, done, info, observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break