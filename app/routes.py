from flask import render_template, request, jsonify
from app import app


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/input', methods=['POST'])
def input():
    print(request.json)
    return jsonify('yes')
