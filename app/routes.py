from flask import render_template, request, jsonify
import numpy as np
from app import app
from app.prediction import regression_predict, cnn_predict


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    result_of_regression = regression_predict(input_data)
    result_of_cnn = cnn_predict(input_data)
    return jsonify(data=[result_of_regression, result_of_cnn])
