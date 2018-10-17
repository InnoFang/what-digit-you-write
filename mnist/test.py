import tensorflow as tf
import numpy as np
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    img = np.invert(Image.open("1.png").convert('L')).ravel().reshape((1, 784))
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(os.path.dirname(__file__), "model/regression-1000.meta"))
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: img}

        regression = graph.get_tensor_by_name("regression:0")

        print(sess.run(regression, feed_dict).flatten().tolist())
