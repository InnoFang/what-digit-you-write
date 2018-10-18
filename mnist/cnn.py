import os
import tensorflow as tf
from mnist.input import mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

learning_rate = 1e-4
n_iterations = 2000
batch_size = 50
dropout = 0.5


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, name):
    return tf.nn.conv2d(
        input=x,
        filter=W,
        strides=[1, 1, 1, 1],
        padding='SAME',
        name=name)


def max_pool_2x2(x, name):
    return tf.nn.max_pool(
        value=x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name=name)


x = tf.placeholder(tf.float32, shape=[None, 784], name="cnn/x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="cnn/y_")
keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

"""
Building the TensorFlow Graph
"""
# 1st layer
W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 28x28 gray image
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="h_conv1")
h_pool1 = max_pool_2x2(h_conv1, "h_pool1")

# 2nd layer
W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="h_conv2")
h_pool2 = max_pool_2x2(h_conv2, "h_pool2")

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
b_fc1 = bias_variable([1024], "b_fc1")
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

# dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, "h_fc1_drop")

# output layer
W_fc2 = weight_variable([1024, 10], "W_fc2")
b_fc2 = bias_variable([10], "b_fc2")
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="y_conv")

"""
Training and Testing
"""
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

if __name__ == '__main__':
    saver = tf.train.Saver()
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
                               os.path.join(os.path.dirname(__file__), 'model/cnn', 'cnn'),
                               global_step=1000)

        print('Saved: ', save_path)
