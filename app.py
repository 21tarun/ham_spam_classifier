import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
transform_model = pickle.load(open('transform.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = transform_model.transform(data).toarray()
        my_prediction = model.predict(vect)





    return render_template('index.html',prediction_text = my_prediction[0])


if __name__ == "__main__":
    app.run(debug=True)