from flask import Flask, request
import urllib.request, json
app = Flask(__name__)
from ml_models.LSTMModel import LSTMTimeSeries
import os
import numpy as np

# model_path = os.getcwd() + '/resources/saved_models/GVS_temperature'
# scaler_path = os.getcwd() + '/resources/scalers/GVS_temperature'
# model = LSTMTimeSeries(model_path, scaler_path)
# model.load_model(model_path, scaler_path)

@app.route('/predict', methods = ['POST'])
def predict():
    event = json.loads(request.data)
    model_name = np.array(event['GraphName'])
    model_path = '{0}\\resources\\saved_models\\{1}'.format(os.getcwd(), str(model_name))
    scaler_path = '{0}\\resources\\scalers\\{1}'.format(os.getcwd(), str(model_name))
    ## Колонка с данными:
    values = np.array(event['SeriesData'])
    model = LSTMTimeSeries(model_path, scaler_path)
    if not os.path.exists(model_name):
        os.mkdir(model_path)
        os.mkdir(scaler_path)
        model.fit(values, epoch = 10)
        model.save_model(model_path, scaler_path)
    res = model.predict_one_step(values[-60:])
    return str(res)


if __name__ == '__main__':
    app.run()
