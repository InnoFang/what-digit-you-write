from app.models.convolutional import LeNet5
from app.models.regresssion import LinearRegression
import os


def predict_by_regression(x_predict):
    model = LinearRegression()

    checkpoint_save_path = 'models/checkpoint/Regression.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        model.load_weights(checkpoint_save_path)

    return model.predict(x_predict).flatten().tolist()


def predict_by_cnn(x_predict):
    model = LeNet5()

    checkpoint_save_path = 'models/checkpoint/LeNet5.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        model.load_weights(checkpoint_save_path)

    return model.predict(x_predict).flatten().tolist()
