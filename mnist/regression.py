import os
import tensorflow as tf
from mnist.input import mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32, name="weights")
b = tf.Variable(tf.zeros([10]), dtype=tf.float32, name="biases")
y = tf.nn.softmax(tf.matmul(x, W) + b)

# train
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    save_path = saver.save(sess,
                           os.path.join(os.path.dirname(__file__), 'model', 'regression.ckpt'),
                           write_meta_graph=False, write_state=False)

    print('Saved: ', save_path)
