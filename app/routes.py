import numpy as np
from app import app
from flask import render_template, request, jsonify
from app.predictors import RegressionPredictor, CNNPredictor


@app.route('/', methods=['GET'])
def index():
    # owing to the Singleton Pattern can make model loading ahead,
    # so let predictors load model before loading page completed.
    RegressionPredictor(), CNNPredictor()

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # reverse (white background, black digit -> black background, white digit)
    # and normalize the image
    input_data = ((255 - np.array(request.json)) / 255.0)

    result_of_regression = RegressionPredictor.predict(input_data)
    result_of_convolutional = CNNPredictor.predict(input_data)
    return jsonify(data=[result_of_regression, result_of_convolutional])
