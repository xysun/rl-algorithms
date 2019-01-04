'''
dqn: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

`observation` from atari is of shape (210,160,3), for CartPole is (4,)

for Breakout, preprocess to 84x84x4
'''
import getopt
import random
from collections import namedtuple, deque

import gym
import numpy as np
import tensorflow as tf

# hyper parameters
REPLAY_MEMORY_SIZE = 200
REPLAY_START_SIZE = 100
BATCH_SIZE = 32
EPISODES = 10
FINAL_EXPLORATION_FRAME = 5
GAMMA = 0.99
EPOCHS_PER_STEP = 1

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state',
                                       'terminal'])  # `terminal` indicates whether `next_state` is terminal


class ExperienceMemory:
    def __init__(self, max_size: int):
        self.experiences = deque()
        self._max_size = max_size

    def store(self, experience: Experience):
        self.experiences.append(experience)
        if len(self.experiences) > self._max_size:
            self.experiences.popleft()

    def size(self):
        return len(self.experiences)

    def sample(self, batch_size: int):
        return random.choices(self.experiences, k=batch_size)


class DQNAgent:
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.experience_memory = ExperienceMemory(REPLAY_MEMORY_SIZE)
        self._classifier = tf.estimator.Estimator(
            model_fn=self.model_fn_nn,
            model_dir="tf_processing/dqn",
            config=tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=False)))

    @staticmethod
    def model_fn_nn(features, labels, mode):
        '''
        Simple feedforward model for CartPole
        '''
        input_layer = tf.reshape(features['x'], [-1, 4])  # hard code 4 is ok since this is only used for CartPole
        dense1 = tf.layers.dense(inputs=input_layer, units=24, activation=tf.nn.relu)
        # dense2 = tf.layers.dense(inputs=dense1, units=8, activation=tf.nn.relu)
        qs = tf.layers.dense(inputs=dense1, units=2)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=qs)

        loss = tf.losses.mean_squared_error(labels=labels, predictions=tf.math.reduce_max(qs, axis=1))
        tf.summary.scalar('loss', loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimiser = tf.train.AdamOptimizer()
            train_op = optimiser.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                              train_op=train_op)

    def predict_q_values(self, state):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': np.reshape(state, (1, self.env.observation_space.shape[0]))}, batch_size=1,
            shuffle=False)
        return self._classifier.predict(input_fn).__next__()

    def action(self, state, epsilon):
        predicted_q_values = self.predict_q_values(state)
        if random.random() <= epsilon:
            return random.choice(range(0, self.env.action_space.n))
        else:
            return np.argmax(predicted_q_values)

    def train(self, x, y):
        input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x}, y=y, batch_size=BATCH_SIZE,
                                                      num_epochs=EPOCHS_PER_STEP,
                                                      shuffle=True)
        self._classifier.train(input_fn, steps=1)


def main(args):
    ops, _ = getopt.getopt(args[1:], '', longopts=['env='])
    env_name = ops[0][1]
    assert env_name in ['CartPole-v1', 'Breakout-v0']
    agent = DQNAgent(env_name)

    # first we accumulate experience under uniformly random policy
    state = agent.env.reset()
    while agent.experience_memory.size() < REPLAY_START_SIZE:
        action = agent.action(state, epsilon=1)
        next_state, reward, is_done, info = agent.env.step(action)
        experience = Experience(state, action, reward, next_state, is_done)
        agent.experience_memory.store(experience)
        state = next_state
        if is_done:
            state = agent.env.reset()

    print("Starting experience collected")
    # then we start training todo why is it so slow
    for i in range(EPISODES):
        state = agent.env.reset()
        is_done = False
        rewards = 0
        if i >= FINAL_EXPLORATION_FRAME:
            epsilon = 0.1
        else:
            epsilon = 1 - 0.9 / FINAL_EXPLORATION_FRAME * i
        while not is_done:
            action = agent.action(state, epsilon=epsilon)
            next_state, reward, is_done, info = agent.env.step(action)
            rewards += reward
            experience = Experience(state, action, reward, next_state, is_done)
            agent.experience_memory.store(experience)
            state = next_state
            # SGD
            batch = agent.experience_memory.sample(BATCH_SIZE)
            # prepare x and y
            x = np.reshape([e.state for e in batch], (BATCH_SIZE, 4))
            y = []
            for e in batch:
                if e.terminal:
                    y.append(e.reward)
                else:
                    y.append(e.reward + GAMMA * np.max(agent.predict_q_values(e.next_state)))
            y = np.asarray(y)
            agent.train(x, y)

        print("Episode %d: total rewards = %d, epsilon = %.2f" % (i, rewards, epsilon))

    # todo then we save the weights and render

    # debug on
    print("Experience replay buffer size:", agent.experience_memory.size())
    print(agent.experience_memory.sample(1))
    # debug off


if __name__ == '__main__':
    tf.app.run()
