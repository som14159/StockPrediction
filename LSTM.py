import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime
import json
# Load the data
df = pd.read_csv('DailyStockPrices.csv')

# Convert the 'date' column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Sort the DataFrame by date
df.sort_values(by='date', inplace=True)

# Extract the target variables
y_open = df['1. open'].values.reshape(-1, 1)
y_high = df['2. high'].values.reshape(-1, 1)
y_close = df['4. close'].values.reshape(-1, 1)
y_volume = df['5. volume'].values.reshape(-1, 1)

# Normalize the target variables
scaler_open = MinMaxScaler(feature_range=(0, 1))
scaler_high = MinMaxScaler(feature_range=(0, 1))
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_volume = MinMaxScaler(feature_range=(0, 1))
y_open_scaled = scaler_open.fit_transform(y_open)
y_high_scaled = scaler_high.fit_transform(y_high)
y_close_scaled = scaler_close.fit_transform(y_close)
y_volume_scaled = scaler_volume.fit_transform(y_volume)

# Define a function to create the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Function to prepare the data for training the LSTM model
def prepare_data_for_lstm(df, window_size=60):
    X = []
    y = []
    for i in range(window_size, len(df)):
        X.append(df['date'].iloc[i-window_size:i].astype(int))  # Convert datetime to integer
        y.append(df['4. close'].iloc[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


# Prepare the data for training
X_open, y_open = prepare_data_for_lstm(df)
X_high, y_high = prepare_data_for_lstm(df)
X_close, y_close = prepare_data_for_lstm(df)
X_volume, y_volume = prepare_data_for_lstm(df)


model_open = create_lstm_model(input_shape=(X_open.shape[1], 1))
model_open.fit(X_open, y_open, epochs=100, batch_size=32)

model_high = create_lstm_model(input_shape=(X_high.shape[1], 1))
model_high.fit(X_high, y_high, epochs=100, batch_size=32)

model_close = create_lstm_model(input_shape=(X_close.shape[1], 1))
model_close.fit(X_close, y_close, epochs=100, batch_size=32)

model_volume = create_lstm_model(input_shape=(X_volume.shape[1], 1))
model_volume.fit(X_volume, y_volume, epochs=100, batch_size=32)


def predict_stock_prices(model, scaler):
    current_date = datetime.now().strftime('%Y-%m-%d')
    X_input = np.array([pd.to_datetime(current_date).timestamp()]).reshape(1, 1, 1)
    prediction_scaled = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0]

predicted_open = int(predict_stock_prices(model_open, scaler_open))
predicted_high = int(predict_stock_prices(model_high, scaler_high))
predicted_close = int(predict_stock_prices(model_close, scaler_close))
predicted_volume = int(predict_stock_prices(model_volume, scaler_volume))

prediction = [{
    'open':predicted_open,  
    'high':predicted_high,  
    'close':predicted_close,  
    'volume':predicted_volume,  
}]

with open("prediction.json",'w') as json_file:
    json.dump(prediction,json_file,indent = 4)
