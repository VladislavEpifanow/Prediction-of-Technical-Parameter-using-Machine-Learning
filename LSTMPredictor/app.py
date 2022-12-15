from flask import Flask, request
import json
app = Flask(__name__)
from LSTMPredictor.ml_models.LSTMModel import LSTMTimeSeries
import os
import numpy as np

TIME_SERIES_MODEL = {}

@app.route('/predict', methods = ['GET'])
def predict():
    event = json.loads(request.data)
    model_name = str(np.array(event['GraphName']))

    if not os.path.exists('{0}\\LSTMPredictor\\resources\\saved_models\{1}'.format(os.getcwd(), model_name)):
        model_name = 'sinus_sensor'
        print('{0}\\LSTMPredictor\\resources\\saved_models\{1}'.format(os.getcwd(), model_name))

    model_path = '{0}\\LSTMPredictor\\resources\\saved_models\\{1}'.format(os.getcwd(), str(model_name))
    scaler_path = '{0}\\LSTMPredictor\\resources\\scalers\\{1}'.format(os.getcwd(), str(model_name))
    ## Колонка с данными:

    values = np.array(event['SeriesData'])
    if model_name not in TIME_SERIES_MODEL:
        model = LSTMTimeSeries(model_path, scaler_path)
        TIME_SERIES_MODEL[model_name] = model
    else:
        model = TIME_SERIES_MODEL[model_name]

    res = model.predict_one_step(values[-60:])
    return str(res)


@app.route('/predict/few', methods = ['GET'])
def predict_twenty_points():
    event = json.loads(request.data)
    model_name = str(np.array(event['GraphName']))
    if not os.path.exists('{0}\\LSTMPredictor\\resources\\saved_models\{1}'.format(os.getcwd(), model_name)):
        model_name = 'sinus_sensor'

    model_path = '{0}\\LSTMPredictor\\resources\\saved_models\\{1}'.format(os.getcwd(), str(model_name))
    scaler_path = '{0}\\LSTMPredictor\\resources\\scalers\\{1}'.format(os.getcwd(), str(model_name))
    ## Колонка с данными:
    values = np.array(event['SeriesData'])
    if model_name not in TIME_SERIES_MODEL:
        model = LSTMTimeSeries(model_path, scaler_path)
        TIME_SERIES_MODEL[model_name] = model
    else:
        model = TIME_SERIES_MODEL[model_name]

    res = model.predict_n_step(values[-60:], 20)
    return res


@app.route('/fit', methods = ['POST'])
def fit():
    event = json.loads(request.data)
    model_name = str(np.array(event['GraphName']))

    model_path = '{0}\\LSTMPredictor\\resources\\saved_models\\{1}'.format(os.getcwd(), str(model_name))
    scaler_path = '{0}\\LSTMPredictor\\resources\\scalers\\{1}'.format(os.getcwd(), str(model_name))
    values = np.array(event['SeriesData'])

    model = LSTMTimeSeries(model_path, scaler_path)
    model.fit(values)


if __name__ == '__main__':
    app.run(host="127.0.0.1")
