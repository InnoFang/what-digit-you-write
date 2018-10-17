from flask import render_template, request, jsonify
from app import app


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print(request.json)
    return jsonify('yes')
