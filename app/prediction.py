import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REGRESSION_MODEL_DIR = os.getcwd() + '/mnist/model/regression/'
REGRESSION_MODEL = 'regression-1000.meta'

CNN_MODEL_DIR = os.getcwd() + '/mnist/model/cnn/'
CNN_MODEL = 'cnn-1000.meta'


def regression_predict(input_data):
    # when you use two model in succession, you need to add the following line of code
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(REGRESSION_MODEL_DIR + REGRESSION_MODEL)
        saver.restore(sess, tf.train.latest_checkpoint(REGRESSION_MODEL_DIR))

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("regression/x:0")
        feed_dict = {x: input_data}

        regression = graph.get_tensor_by_name("y:0")
        return sess.run(regression, feed_dict).flatten().tolist()


def cnn_predict(input_data):
    # when you use two model in succession, you need to add the following line of code
    tf.reset_default_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(CNN_MODEL_DIR + CNN_MODEL)
        saver.restore(sess, tf.train.latest_checkpoint(CNN_MODEL_DIR))

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("cnn/x:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        feed_dict = {x: input_data, keep_prob: 1.0}

        y_conv = graph.get_tensor_by_name("y_conv:0")
        return sess.run(y_conv, feed_dict).flatten().tolist()
