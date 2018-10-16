from flask import render_template, request, jsonify
from app import app


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/input')
def input():
    print(request.json)
