import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, l_rate=1e-2):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            l1 = tf.layers.dense(inputs=self._X, units=100, activation=tf.nn.tanh)
            l2 = tf.layers.dense(inputs=l1, units=100, activation=tf.nn.tanh)

            self._Qpred = tf.layers.dense(inputs=l2, units=self.output_size)
            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
            self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
            # Saver Must be defined after all variables defined
            self.saver = tf.train.Saver()

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train],
                                feed_dict={self._X: x_stack, self._Y: y_stack})

    def save(self):
        save_path = self.saver.save(self.session, "data/" + self.net_name + "model.ckpt")
        print("Model saved : %s" % save_path)

    def restore(self):
        self.saver.restore(self.session, "data/" + self.net_name + "model.ckpt")
# Class DQN end
