from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('linreg.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        bedrooms = int(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        prediction = model.predict([[bedrooms, bathrooms]])
        return render_template('index.html', prediction_text="The prediction is {}".format(prediction[0]))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)