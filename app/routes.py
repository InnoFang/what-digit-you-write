import numpy as np
from app import app
from flask import render_template, request, jsonify
from app.prediction import predict_by_regression, predict_by_cnn


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # normalized - normalize the data from 0 to 255 to 0 to 1.
    input_data = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0)
    result_of_regression = predict_by_regression(input_data.reshape(1, 784))
    result_of_convolutional = predict_by_cnn(input_data.reshape(1, 28, 28, 1))
    return jsonify(data=[result_of_regression, result_of_convolutional])
