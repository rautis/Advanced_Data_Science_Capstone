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

model = load_model("models/lstm-weather-data-look-back-1.h5")
graph = tf.get_default_graph()

scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

class AirQualityIndex(Resource):
    
    def get(self):
        return { "status": "success", "message": "Use POST API to pass the previous hour data" }
    
    def post(self):
        #Mannerheimintie	Air pressure (msl) (hPa)	
        #Relative humidity (%)	Rain intensity (mm/h)	
        #Air temperature (degC)	Wind direction (deg)	
        #Wind speed (m/s)
        global graph
        with graph.as_default():
            X = np.array([[
                request.json["paqi-1"],
                request.json["ap-1"],
                request.json["rh-1"],
                request.json["ri-1"],
                request.json["at-1"],
                request.json["wd-1"],
                request.json["ws-1"],
            ],
            [
                request.json["paqi-2"],
                request.json["ap-2"],
                request.json["rh-2"],
                request.json["ri-2"],
                request.json["at-2"],
                request.json["wd-2"],
                request.json["ws-2"],
            ]])
            
            X = scaler_X.transform(X)
            X = X.reshape(1,2,7)

            yhat = scaler_y.inverse_transform(model.predict(X))

            return { "aqi": yhat.tolist() }

api.add_resource(AirQualityIndex, '/airquality') 

if __name__ == '__main__':
     app.run(port='5001')