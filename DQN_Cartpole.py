# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
from gym.envs.registration import register
import math
import DQN as dqn

register(
    id='CartPole-v1565',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    # 'wrapper_config.TimeLimit.max_episode_steps' limits maximum step
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10001},
    reward_threshold=-100
)

env = gym.make('CartPole-v1565')

# input_size = 4, output_size = 2
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995


def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1.0 - math.log10((t+1) * epsilon_decay)))


def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break


def main():
    max_episodes = 5000
    replay_buffer = deque()
    epsilon = 1.0

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        steps = []
        for episode in range(max_episodes):
            e = get_epsilon(episode)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    # popleft : return left value and pop it
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
                if step_count > 10000:
                    break
            print("Episode: {} steps: {}".format(episode, step_count))

            steps.append(step_count)
            # if recent 10 episodes' steps mean > 200, break -> success
            if len(steps) > 10:
                steps.pop(0)
                if np.mean(steps, axis=0) > 200:
                    break

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                print("Loss: ", loss)
                sess.run(copy_ops)
        mainDQN.save()
        targetDQN.save()

        for _ in range(10):
            bot_play(mainDQN)


if __name__ == "__main__":
    main()