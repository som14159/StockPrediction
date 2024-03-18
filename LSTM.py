import pandas as pd
df = pd.read_csv('DailyStockPrices.csv')
print(df.head())
df = df.loc[::-1].reset_index(drop=True)
print(df.head())
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

X = df.index.values.reshape(-1, 1)  # Reshape to make it 2D
y = df[['1. open', '2. high', '4. close', '3. low', '5. volume']].values
print(X)
print(y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0000000000001, random_state=42)

# Scale the features
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the target variable
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Reshape input data for LSTM
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[2])))
model.add(Dense(5))  
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print('Test loss:', loss)

# Make predictions
predictions = model.predict(X_test_scaled)

predictions = scaler_y.inverse_transform(predictions)

print("Predictions:")
for i in range(len(predictions)):
    print("Predicted:", predictions[i])
    print("Actual:", y_test[i])

tommorow=[[100]]
tommorow = scaler_X.transform(tommorow)
tommorow = tommorow.reshape((tommorow.shape[0], 1, tommorow.shape[1]))
predictions = model.predict(tommorow)
predictions = scaler_y.inverse_transform(predictions)
print("Tommorows Predictions: ",predictions)

X = np.concatenate((X, [[100]]))
y = np.concatenate((y, predictions))

# Plot updated values
fig, axs = plt.subplots(5, 1, figsize=(12, 10))

for i, ax in enumerate(axs.flat):
    if i < 5:
        ax.plot(X, y[:, i], label=df.columns[i+1])
        ax.set_xlabel('Day')
        ax.set_ylabel(df.columns[i+1])
        ax.legend()

plt.show()

import json

# Extract predictions
open_value, high_value, close_value, low_value, volume_value = predictions.flatten()

# Create data dictionary
data = {
    "open": float(open_value),
    "high": float(high_value),
    "close": float(close_value),
    "low": float(low_value),
    "volume": float(volume_value)
}

# File path
file_path = "StockData.json"

# Write data to the JSON file
with open(file_path, "w") as json_file:
    json.dump(data, json_file)

print("Data has been written to 'data.json'.")
