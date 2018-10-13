import os
import tensorflow as tf
from mnist.input import mnist
from mnist import model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

learning_rate = 1e-4
n_iterations = 2000
batch_size = 50
dropout = 0.5

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(dtype=tf.float32)
y_conv, variables = model.convolutional(x, keep_prob)

"""
Training and Testing
"""
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_iterations):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            print("Step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout})

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    save_path = saver.save(sess,
                           os.path.join(os.path.dirname(__file__), 'model', 'cnn.ckpt'),
                           write_meta_graph=False, write_state=False)

    print('Saved: ', save_path)
