from tensorflow.keras.models import load_model


class RegressionPredictor(object):
    __model = None

    # ensure predictor be initialized only once by using Singleton Pattern
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '__instance'):
            cls.__instance = super().__new__(cls)
            cls.__model = load_model('app/models/regression.h5')
        return cls.__instance

    @staticmethod
    def predict(input_data):
        assert RegressionPredictor.__model, \
            "Use 'RegressionPredictor()' to initialize before using 'RegressionPredictor.predict()'"

        x = input_data.reshape(1, 28, 28)
        return RegressionPredictor.__model.predict(x).flatten().tolist()


class CNNPredictor(object):
    __model = None

    # ensure predictor be initialized only once by using Singleton Pattern
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '__instance'):
            cls.__instance = super().__new__(cls)
            cls.__model = load_model('app/models/convolutional.h5')
        return cls.__instance

    @staticmethod
    def predict(input_data):
        assert CNNPredictor.__model, "Use 'CNNPredictor()' to initialize before using 'CNNPredictor.predict()'"

        x = input_data.reshape(1, 28, 28, 1)
        return CNNPredictor.__model.predict(x).flatten().tolist()
