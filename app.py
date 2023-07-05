import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = None
scaler = None

@app.before_first_request
def load_model():
    global model, scaler
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = request.form.to_dict()
    input_values = [float(data[key]) for key in data]
    input_scaled = scaler.transform([input_values])
    prediction = model.predict(input_scaled)

    output = round(prediction[0], 3)

    return render_template('index.html', prediction_text='Estimated Range: {} km'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
