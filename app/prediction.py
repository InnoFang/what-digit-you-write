import tensorflow as tf
import numpy as np
from mnist import model

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 32])

y_regression, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, 'model/regression.ckpt')

keep_prob = tf.placeholder(tf.float32)
y_cnn, variables = model.cnn(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, 'model/cnn.ckpt')


def regression(data):
    return sess.run(y_regression, feed_dict={x: data})


def cnn(data):
    return sess.run(y_cnn, feed_dict={x: data, keep_prob: 1.0})