'''
[DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) for CartPole env, Atari env (maybe) upcoming

Notes:
- `observation` from atari is of shape (210,160,3), for CartPole is (4,)
- for Breakout, preprocess to 84x84x4

Architecture:
- for CartPole, 2 layers of feedforward with 24 units each

To run on a fresh instance (with numpy + tensorflow + gym installed)

```
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
TF_CPP_MIN_LOG_LEVEL="3" python3 dqn.py --env=CartPole-v1
```
'''
import getopt
import random
import sys
from collections import deque, namedtuple

import gym
import numpy as np
import tensorflow as tf

# hyper parameters
REPLAY_MEMORY_SIZE = 100000
REPLAY_START_SIZE = 64
BATCH_SIZE = 64
EPISODES = 1000
EPSILON_DECAY = 0.96
STARTING_EPSILON = 1
FINAL_EPSILON = 0.05
GAMMA = 0.99
EPOCHS_PER_STEP = 1
MODEL_DIR = 'tf_processing/dqn'
VALIDATION_DIR = 'tf_processing/dqn_validation'

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'terminal'])


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
        return random.sample(self.experiences, k=batch_size)


class DQNAgent:
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.experience_memory = ExperienceMemory(REPLAY_MEMORY_SIZE)
        self._classifier = tf.estimator.Estimator(
            model_fn=self.model_fn_nn,
            model_dir=MODEL_DIR,
            config=tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=False)))

        if env_name == 'CartPole-v1':
            self.OBSERVATION_SPACE_N = 4
        else:
            self.OBSERVATION_SPACE_N = None

    @staticmethod
    def model_fn_nn(features, labels, mode):
        '''
        Simple feedforward model for CartPole
        '''
        input_layer = tf.reshape(features['x'], [-1, 4])  # hard code 4 is ok since this is only used for CartPole
        dense1 = tf.layers.dense(inputs=input_layer, units=24, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=24, activation=tf.nn.relu)
        qs = tf.layers.dense(inputs=dense2, units=2)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=qs)

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = tf.losses.mean_squared_error(labels=labels, predictions=qs)
            tf.summary.scalar('loss', loss)
            optimiser = tf.train.AdamOptimizer(learning_rate=0.0005)
            train_op = optimiser.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                              train_op=train_op)

    def predict_q_values(self, states):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': states}, batch_size=states.shape[0],
            shuffle=False)
        return self._classifier.predict(input_fn)

    def action(self, state, epsilon):
        q_values = list(self.predict_q_values(np.reshape(state, (1, self.OBSERVATION_SPACE_N))))
        assert len(q_values) == 1
        if random.random() <= epsilon:
            return random.choice(range(0, self.env.action_space.n))
        else:
            assert q_values[0].shape[0] == 2
            return np.argmax(q_values[0])

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
        reward_to_record = -999 if is_done else reward
        experience = Experience(state, action, reward_to_record, next_state, is_done)
        agent.experience_memory.store(experience)
        state = next_state
        if is_done:
            state = agent.env.reset()

    print("Starting experience collected")
    epsilon = STARTING_EPSILON
    # then we start training
    for i in range(1, EPISODES + 1):
        state = agent.env.reset()
        is_done = False
        rewards = 0

        if i % 10 == 0:
            # validation episode
            total_rewards = 0
            for j in range(100):
                state = agent.env.reset()
                is_done = False
                rewards = 0

                while not is_done:
                    action = agent.action(state, epsilon=FINAL_EPSILON)
                    next_state, reward, is_done, info = agent.env.step(action)
                    rewards += reward
                    state = next_state

                total_rewards += rewards
            avg_reward = total_rewards / 5.
            summary_writer = tf.summary.FileWriter(VALIDATION_DIR)
            summary = tf.Summary()
            summary.value.add(tag='validation_reward', simple_value=avg_reward)
            summary_writer.add_summary(summary, global_step=i)
            summary_writer.flush()

            print("Validation episode %d avg reward %.2f" % (i, avg_reward))
            epsilon = max(FINAL_EPSILON, epsilon * EPSILON_DECAY)
            if avg_reward > 195:
                print("Success! Stopping..")
                sys.exit()

        else:

            while not is_done:
                action = agent.action(state, epsilon=epsilon)
                next_state, reward, is_done, info = agent.env.step(action)
                rewards += reward

                reward_to_record = -999 if is_done else reward
                experience = Experience(state, action, reward_to_record, next_state, is_done)

                agent.experience_memory.store(experience)
                state = next_state
                # SGD
                batch = agent.experience_memory.sample(BATCH_SIZE)
                # prepare x and y
                # t = time.monotonic()
                x = np.reshape([e.state for e in batch], (BATCH_SIZE, agent.OBSERVATION_SPACE_N))
                next_states = np.reshape([e.next_state for e in batch], (BATCH_SIZE, agent.OBSERVATION_SPACE_N))
                next_states_q_values = list(agent.predict_q_values(next_states))
                curr_state_q_values = list(agent.predict_q_values(x))
                assert (len(next_states_q_values) == BATCH_SIZE)
                assert (len(curr_state_q_values) == BATCH_SIZE)
                y = []
                for idx, e in enumerate(batch):
                    next_qs = next_states_q_values[idx]
                    curr_qs = curr_state_q_values[idx]
                    # debug to make sure batch predict preserves input order
                    # print(next_qs)
                    # print(next(agent.predict_q_values(np.reshape(e.next_state, (1,4)))))
                    # print(curr_qs)
                    # print(next(agent.predict_q_values(np.reshape(e.state, (1, 4)))))

                    # IMPORTANT!
                    if e.terminal:
                        curr_qs[e.action] = e.reward
                    else:
                        curr_qs[e.action] = e.reward + GAMMA * np.max(next_qs)
                    y.append(curr_qs)
                y = np.reshape(np.asarray(y), (BATCH_SIZE, 2))
                # print("====== step")
                agent.train(x, y)

            print("Episode %d: total rewards = %d, epsilon = %.2f, size of replay memory %d" % (
                i, rewards, epsilon, agent.experience_memory.size()))
            with open('logs/dqn.csv', 'a') as f:
                f.write("%d,%d,%.2f\n" % (i, rewards, epsilon))


if __name__ == '__main__':
    tf.app.run()
