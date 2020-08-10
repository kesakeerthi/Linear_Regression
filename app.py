# -*- coding: utf-8 -*-
"""
Created on Sat Aug 1 2020
@author: Sai Keerthi
"""
import pickle
import pandas as pd
from flask import Flask, request
from flask_cors import CORS,cross_origin
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open("scaler.pkl", "rb")
scaler = pickle.load(pickle_in)

pickle_in = open("model.pkl", "rb")
regressor = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome to Undergraduate Admission Predictor"

@app.route('/predict', methods=["GET"])
@cross_origin()
def predict():
    
    """Let's predict the percentage score for Undergraduate Admissions in USA.
       Single Entity
    ---
    parameters:  
      - name: GRE Score
        in: query
        type: number
        required: true
      - name: TOEFL Score
        in: query
        type: number
        required: true
      - name: University Rating
        in: query
        type: number
        required: true
      - name: SOP
        in: query
        type: number
        required: true
      - name: LOR
        in: query
        type: number
        required: true
      - name: CGPA
        in: query
        type: number
        required: true
      - name: Research
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """

    GRE_Score          = request.args.get("GRE Score")
    TOEFL_Score        = request.args.get("TOEFL Score")
    University_Rating  = request.args.get("University Rating")
    SOP                = request.args.get("SOP")
    LOR                = request.args.get("LOR")
    CGPA               = request.args.get("CGPA")
    Research           = request.args.get("Research")

    a = regressor.predict(scaler.transform([[GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]]))
    res = a[0] * 100
    result = round(res,1)
    print(result)

    return "Hello, The chance of getting Admission is " + str(result) + ' %'

@app.route('/predict_file', methods=["POST"])
@cross_origin()
def predict_file():
    """Let's predict the percentages score of all students(in CSV) for Undergraduate Admissions in USA.
       Bulk Predict from a CSV File
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output
        
    """

    df_file = pd.read_csv(request.files.get("file"))
    print(df_file.head())
    prediction = regressor.predict(scaler.transform(df_file))
    res = list(prediction)
    result = []
    for i in res:
        temp = i * 100
        result.append(round(temp,1))

    return "Hello, The chances of getting Admission in percentages is " + str(result)

if __name__=='__main__':
    #to run on cloud
	app.run() # running the app

    #to run locally
    #app.run(host='127.0.0.1', port=8000, debug=True)
