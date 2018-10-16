import tensorflow as tf


# Softmax Regression Model
def regression(x):
    W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32, name="weights")
    b = tf.Variable(tf.zeros([10]), dtype=tf.float32, name="biases")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y, [W, b]


# Multilayer Convolutional Network
def cnn(x, keep_prob):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(value=0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(
            input=x,
            filter=W,
            strides=[1, 1, 1, 1],
            padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(
            value=x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')

    """
    Building the TensorFlow Graph
    """
    # 1st layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # 28x28 gray image
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
