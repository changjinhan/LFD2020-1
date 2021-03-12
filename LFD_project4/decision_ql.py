from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import random


class QLearningDecisionPolicy:
    def __init__(self, epsilon, gamma, decay, lr, actions, input_dim, model_dir):
        # select action function hyperparameter
        self.epsilon = epsilon
        # q functions hyperparameter
        self.gamma = gamma
        # neural network hyperparmeter
        self.lr = lr
        self.decay = decay
        self.counter = 0
        self.actions = actions
        output_dim = len(actions)

        # neural network input and output place holder
        self.x = tf.compat.v1.placeholder(tf.float32, [None, input_dim])
        self.y = tf.compat.v1.placeholder(tf.float32, [output_dim])
        self.training = tf.placeholder_with_default(True, shape=())

        # 2-layer fully connected network
        # fc = tf.layers.Dropout(0.2)(self.x, training= self.training)
        fc = tf.layers.dense(self.x, 64, activation=tf.nn.relu)
        # fc = tf.layers.Dropout(0.2)(fc, training= self.training)
        self.q = tf.layers.dense(fc, output_dim)

        # loss
        loss = tf.square(self.y - self.q)

        # train operation
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

        # session
        self.sess = tf.compat.v1.Session()

        # initalize variables
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)

        # saver
        self.saver = tf.compat.v1.train.Saver()

        # restore model
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("load model: %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def select_action(self, current_state, is_training=True):
        if random.random() >= self.epsilon or not is_training:
            action_q_vals = self.sess.run(self.q, feed_dict={
                self.x: current_state})  # , self.training : is_training}) # 현 state에서 가능한 모든 action의 value를 구함
            action_idx = np.argmax(action_q_vals)
            action = self.actions[action_idx]

        else:  # randomly select action
            action = self.actions[random.randint(0, len(self.actions) - 1)]

        return action

    def update_q(self, current_state, action, reward, next_state):
        # Q(s, a)
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
        # Q(s', a')
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        # a' index
        next_action_idx = np.argmax(next_action_q_vals)
        # create target
        action_q_vals[0, self.actions.index(action)] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        # delete minibatch dimension
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: current_state, self.y: action_q_vals})
        # decay epsilon
        self.counter += 1
        if self.epsilon > 0.1 and self.counter % 1000 == 0:
            self.epsilon = self.decay * self.epsilon
            print("epsilon : {}".format(self.epsilon))

    def save_model(self, output_dir):
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        checkpoint_path = output_dir + '/model'
        self.saver.save(self.sess, checkpoint_path)

