# -*- coding: utf-8 -*-
#
# Same as DQN_Cartpole.py, but This uses saved DQN model.
#
import tensorflow as tf
import numpy as np
import gym
from gym.envs.registration import register
import DQN as dqn

register(
    id='CartPole-v1565',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10001},
    reward_threshold=-100
)

env = gym.make('CartPole-v1565')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n


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
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        mainDQN.restore()
        for _ in range(10):
            bot_play(mainDQN)


if __name__ == "__main__":
    main()