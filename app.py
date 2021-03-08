# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:51:32 2021

@author: ab522tx
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('knn_model.pkl', 'rb'))

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', 
                           prediction_text='Predicted for Iris Featues {} Iris Class is {}'.format(final_features,output))


if __name__ == "__main__":
    app.run(debug=True)