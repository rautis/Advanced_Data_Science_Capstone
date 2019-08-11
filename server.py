from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import joblib

app = Flask(__name__)
api = Api(app)

model = load_model("models/lstm-filter-data-look-back-8.h5")
graph = tf.get_default_graph()

scaler_X = joblib.load("models/scaler_X_filter.pkl")
scaler_y = joblib.load("models/scaler_y_filter.pkl")

class AirQualityIndex(Resource):
    
    def get(self):
        return { "status": "success", "message": "Use POST API to pass the previous hour data" }
    
    def post(self):
        
        global graph
        with graph.as_default():

            # Get input data
            data = request.json["data"]
            entris = []

            for d in data:
                entris.append(d["paqi"])
                entris.append(d["rm"])
                entris.append(d["rstd"])
                entris.append(d["wiener"])

            X = np.array(entris).reshape(8,4)            
            X = scaler_X.transform(X)
            X = X.reshape(1,8,4)

            yhat = scaler_y.inverse_transform(model.predict(X))

            return { "aqi": yhat.tolist() }

api.add_resource(AirQualityIndex, '/airquality') 

if __name__ == '__main__':
     app.run(port='5001')