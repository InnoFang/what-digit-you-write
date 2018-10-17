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
    return jsonify(result=[regression_predict(input_data), cnn_predict(input_data)])
