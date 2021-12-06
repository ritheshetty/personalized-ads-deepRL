import tensorflow as tf
import numpy as np
from ..envs import adserver
import random
import argparse
from collections import deque
import gym
from gym import logger


class DqnAgent:

    def __init__(self, state_size, action_size, num_ads):
        self.scenario_name = "DQN Agent"
        self.num_ads = num_ads
        self.gamma = 0.95
        self.epsilon = 0.2
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.state_size = state_size
        self.action_size = action_size
        self.q_net = self.build_dqn_model()
        self.target_q_net = self.build_dqn_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = tf.keras.backend.abs(error) <= clip_delta

        squared_loss = 0.5 * tf.keras.backend.square(error)
        quadratic_loss = 0.5 * tf.keras.backend.square(clip_delta) + clip_delta * (tf.keras.backend.abs(error) - clip_delta)

        return tf.keras.backend.mean(tf.where(cond, squared_loss, quadratic_loss))

    def build_dqn_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=tf.keras.optimizers.Adam(lr=0.0025))
        return model

    def random_policy(self, state):
        return np.random.randint(0, self.action_size)

    def collect_policy(self, state):
        if np.random.random() < self.epsilon:
            return self.random_policy(state)
        return self.policy(state)

    def create_input(self, state):
        state_input = np.array([])
        ads, impressions, clicks = state
        for i in range(self.num_ads):
            state_input = np.append(state_input, ads[i].clicks)
        for i in range(self.num_ads):
            state_input = np.append(state_input, ads[i].impressions)
        state_input = np.append(state_input, clicks)
        state_input = np.append(state_input, impressions)
        state_input = np.reshape(state_input, (1, -1))
        state_input = tf.convert_to_tensor(state_input, dtype=tf.float32)
        return state_input

    def policy(self, state):
        state_input = self.create_input(state)
        action_q = self.q_net.predict(state_input)[0]
        action = np.argmax(action_q, axis=0)
        return action

    def update_target_network(self):
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, batch):
        state_batch, next_state_batch, action_batch, reward_batch, done_batch \
            = batch

        for state, next_state, action, reward, done in zip(state_batch, next_state_batch, action_batch, reward_batch, done_batch):
            state_input = self.create_input(state)
            current_q = self.q_net.predict(state_input)
            target_q = np.copy(current_q)
            next_state_input = self.create_input(next_state)
            target_q_val = reward
            if not done:
                    target_q_val += self.gamma * np.max(self.target_q_net.predict(next_state_input)[0])
            target_q[0][action] = target_q_val
            training_history = self.q_net.fit(x=state_input, y=target_q, verbose=0)
            loss = training_history.history['loss']
            return loss


class ReplayBuffer:

    def __init__(self):
        self.experiences = deque(maxlen=5000)

    def store_experience(self, state, next_state, reward, action, done):
        self.experiences.append((state, next_state, reward, action, done))

    def sample_batch(self):
        batch_size = 32
        sampled_batch = random.sample(self.experiences, batch_size)
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for experience in sampled_batch:
            state_batch.append(experience[0])
            next_state_batch.append(experience[1])
            reward_batch.append(experience[2])
            action_batch.append(experience[3])
            done_batch.append(experience[4])
        return np.array(state_batch), np.array(next_state_batch), np.array(
            action_batch), np.array(reward_batch), np.array(done_batch)


def collect_experiences(env, agent, buffer, impressions):
    state = env.reset(agent.scenario_name)
    for time in range(impressions):
        env.render()
        action = agent.collect_policy(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -10.0
        buffer.store_experience(state, next_state, reward, action, done)
        state = next_state


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--impressions', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    time_series_frequency = args.impressions // 10

    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)
    env.seed(args.seed)
    state_size = (2 * args.num_ads) + 2
    action_size = env.action_space.n
    agent = DqnAgent(state_size, action_size, args.num_ads)
    buffer = ReplayBuffer()

    reward = 0
    done = False
    num_episodes = 1000
    for episode_cnt in range(num_episodes):
        collect_experiences(env, agent, buffer, args.impressions)
        experience_batch = buffer.sample_batch()
        loss = agent.train(experience_batch)
        print("LOSS:", loss)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        if episode_cnt % 10 == 0:
            agent.update_target_network()
    env.close()
    print('DONE')


run_model()
