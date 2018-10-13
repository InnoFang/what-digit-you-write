import os
import tensorflow as tf
from mnist.input import mnist
from mnist import model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
Defining the Constant
"""
n_iterations = 2000
batch_size = 50


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
y, variables = model.regression(x)

"""
Training and Testing
"""
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(n_iterations):
        batch = mnist.train.next_batch(batch_size)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    save_path = saver.save(sess,
                           os.path.join(os.path.dirname(__file__), 'model', 'regression.ckpt'),
                           write_meta_graph=False, write_state=False)

    print('Saved: ', save_path)
