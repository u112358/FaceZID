# MIT License
#
# Copyright (c) 2017 BingZhang Hu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
from util import net_builder as nb
from util import data_reader as dr
import tensorflow.contrib.slim as slim
import argparse
import sys
from configurer import Configurer

class FaceZID():
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, sess, config):
        self.sess = sess
        self.root_dir = os.getcwd()
        self.subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(os.path.expanduser('logs'), self.subdir)
        self.model_dir = os.path.join(os.path.expanduser('models'), self.subdir)
        self.learning_rate = 0.1
        self.batch_size = 60
        self.class_num = 2000
        self.max_epoch = 20
        self.data_dir = config.data_dir
        self.image_in = tf.placeholder(tf.float32, [self.batch_size, 250, 250, 3])
        self.label_in = tf.placeholder(tf.float32, [self.batch_size])
        self.net = self._build_net()
        self.loss = self._build_loss()
        self.accuracy = self._build_accuracy()
        self.opt = tf.train.AdamOptimizer(self.learning_rate,beta1=0.9, beta2=0.999, epsilon=0.1).minimize(self.loss)

    def _build_net(self):
        # convolution layers
        net, _ = nb.inference(images=self.image_in, keep_probability=1.0, bottleneck_layer_size=128, phase_train=True,
                              weight_decay=0.0)

        # with tf.variable_scope('output') as scope:
        #     weights = tf.get_variable('weights', [1024, self.class_num], dtype=tf.float32,
        #                               initializer=tf.truncated_normal_initializer(stddev=1e-2))
        #     biases = tf.get_variable('biases', [self.class_num], dtype=tf.float32, initializer=tf.constant_initializer())
        #     output = tf.add(tf.matmul(net, weights), biases, name=scope.name)
        #     nb.variable_summaries(weights,'weights')
        #     nb.variable_summaries(biases,'biases')
        logits = slim.fully_connected(net, self.class_num, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.0), scope='logits', reuse=False)
        return logits

    def _build_loss(self):
        label_int64 = tf.cast(self.label_in,tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.net, labels=label_int64, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        tf.summary.scalar('loss', loss)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([loss] + regularization_losses, name='total_loss')
        return total_loss

    def _build_accuracy(self):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.net, 1),tf.cast(self.label_in,tf.int64))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def train(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        saver = tf.train.Saver()
        data_reader = dr.DataReader(self.data_dir, 163446, self.batch_size, 0.8, reproducible=True)
        tf.summary.image('image', self.image_in, 10)
        summary_op = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        writer_test = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)
        step = 1
        while data_reader.epoch < self.max_epoch:
            if step % 100 == 0:
                images, label = data_reader.next_batch(phase_train=False)
                reshaped_image = np.reshape(images, [self.batch_size, 250, 250, 3])
                feed_dict = {self.image_in: reshaped_image, self.label_in: label}
                start_time = time.time()
                err, acc, sum = self.sess.run([self.loss, self.accuracy, summary_op], feed_dict=feed_dict)
                duration = time.time() - start_time
                print('Epoch:%d/%d\tTime:%.3f\tLoss:%2.4f\tAcc:%2.4f\t@[TEST]' % (
                    data_reader.current_test_batch_index, data_reader.epoch, duration, err, acc))
                writer_test.add_summary(sum, step)
            else:
                images, label = data_reader.next_batch(phase_train=True)
                reshaped_image = np.reshape(images, [self.batch_size, 250, 250, 3])
                feed_dict = {self.image_in: reshaped_image, self.label_in: label}
                start_time = time.time()
                err, acc, sum, _ = self.sess.run([self.loss, self.accuracy, summary_op, self.opt], feed_dict=feed_dict)
                duration = time.time() - start_time
                print('Epoch:%d/%d\tTime:%.3f\tLoss:%2.4f\tAcc:%2.4f\t' % (
                    data_reader.current_train_batch_index, data_reader.epoch, duration, err, acc))
                writer_train.add_summary(sum, step)
            if step % 3268 == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(self.sess, self.model_dir, step)

            test


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--workplace', type=bool,
                        help='where the code runs', default=False)

    return parser.parse_args(argv)

if __name__ == '__main__':
    config = Configurer(parse_arguments(sys.argv[1:]).workplace)
    gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    this_session = tf.Session(config=gpu_config)
    model = FaceZID(this_session,config)
    model.train()
