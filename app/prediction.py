import tensorflow as tf


def predict_by_regression(x_predict):
    model = tf.keras.models.load_model('app/models/regression.h5')
    return model.predict(x_predict).flatten().tolist()


def predict_by_cnn(x_predict):
    model = tf.keras.models.load_model('app/models/convolutional.h5')
    return model.predict(x_predict).flatten().tolist()
