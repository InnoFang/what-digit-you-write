import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
import os


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(5, 5), strides=1, padding='same', activation='relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        self.c2 = Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding='same', activation='relu')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        self.flatten = Flatten()
        # self.f3 = Dense(1024, activation='relu')
        # self.d3 = Dropout(0.2)
        self.f3 = Dense(120, activation='relu')
        self.f4 = Dense(84, activation='relu')
        self.f5 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        # x = self.f3(x)
        # x = self.d3(x)
        x = self.f3(x)
        x = self.f4(x)
        y = self.f5(x)
        return y


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = LeNet5()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = 'checkpoint/LeNet5.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    model.fit(x_train, y_train,
                   batch_size=32,
                   epochs=5,
                   validation_data=(x_test, y_test),
                   validation_freq=1,
                   callbacks=[cp_callback])

    model.summary()