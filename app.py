import numpy as np
from flask import Flask, app, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    test = model.predict(features)
    return render_template('index.html', prediction_text = 'Salary should be approximately ${}'.format(test))


if __name__ == "__main__":
    app.run()