import numpy as np
from flask import Flask, app, json, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
companies = list(model.threshold_dict['Company'])
locations = list(model.threshold_dict['Location'])
titles = list(model.threshold_dict['Job_Title'])
subspecialties = list(model.threshold_dict['Subspecialty'])
roles = list(model.threshold_dict['Role'])

@app.route('/')
def home():
    return render_template('index.html', companies = companies, locations=locations,
                            titles = titles, subspecialties = subspecialties, roles = roles)

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    test = model.predict(features)
    test = format(int(test),',')
    return render_template('index.html', prediction_text = 'Approximate Salary: ${}'.format(test),
                            companies = companies, locations=locations, titles = titles,
                            subspecialties = subspecialties, roles = roles)


if __name__ == "__main__":
    app.run()