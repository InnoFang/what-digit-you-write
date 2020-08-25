import numpy as np
from app import app
from flask import render_template, request, jsonify
from app.predictors import predict_by_regression, predict_by_cnn


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # normalized
    input_data = ((255 - np.array(request.json)) / 255.0)

    result_of_regression = predict_by_regression(input_data)
    result_of_convolutional = predict_by_cnn(input_data)
    return jsonify(data=[result_of_regression, result_of_convolutional])
