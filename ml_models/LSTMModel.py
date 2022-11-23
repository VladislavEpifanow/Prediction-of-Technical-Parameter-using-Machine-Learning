from keras.layers import *
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
import joblib


class LSTMTimeSeries:
    def __init__(self, model_name, scaler_name, prev_features=60):
        self.model_name = model_name
        self.prev_features = prev_features
        try:
            self.load_model(model_name, scaler_name)
        except:
            print("Models wasn't found.Creating new")
            # Инициализация Рекурентной нейронной сети
            self.model = Sequential()
            self.model.add(LSTM(units=50, return_sequences=True, input_shape=(prev_features, 1)))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50))
            self.model.add(Dropout(0.2))
            # Выходной слой
            self.model.add(Dense(units=1))
            self.model.summary()

    def fit(self, time_series, optimizer='adam', epoch=500, batch_size=16):
        self.sc = MinMaxScaler(feature_range=(0, 1))
        ts_scaled = self.sc.fit_transform(time_series.reshape(-1, 1))

        ## Data Preparing
        X_train = []
        y_train = []

        window = self.prev_features

        for i in range(window, time_series.shape[0]):
            X_train_ = np.reshape(ts_scaled[i - window:i, 0], (window, 1))
            X_train.append(X_train_)
            y_train.append(ts_scaled[i, 0])
        X_train = np.stack(X_train)
        y_train = np.stack(y_train)

        ## Fitting

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model.fit(X_train, y_train, epochs=epoch, batch_size=16)

    def predict_one_step(self, data):
        res = self.model.predict(self.sc.transform(data.reshape(-1, 1)).reshape(1, 60, 1))
        res = self.sc.inverse_transform(res)
        return res[0][0]

    def predict_n_step(self, n_step):
        pass

    def load_model(self, model_path, scaler_path):
        self.model = load_model(model_path)
        self.sc = joblib.load(scaler_path)

    def save_model(self, model_path, scaler_path):
        self.model.save(model_path)
        joblib.dump(self.sc, scaler_path)
