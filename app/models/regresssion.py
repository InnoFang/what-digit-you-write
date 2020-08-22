import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Activation


class LinearRegression(Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.flatten = Flatten()
        self.W = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
        self.b = tf.Variable(tf.random.truncated_normal([10], stddev=0.1))
        self.a = Activation('softmax')

    def call(self, x, **kwargs):
        x = self.flatten(x)
        y = self.a(tf.matmul(x, self.W) + self.b)
        return y


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = LinearRegression()

    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = 'checkpoint/Regression.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=20,
              validation_data=(x_test, y_test),
              validation_freq=1,
              callbacks=[cp_callback])

    model.summary()