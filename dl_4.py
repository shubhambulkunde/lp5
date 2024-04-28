import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load training data
train_df = pd.read_csv(r'Google_Stock_Price_Train.csv')
test_df = pd.read_csv(r'Google_Stock_Price_Test.csv')

# Convert 'Close' column to float type and remove commas
train_df['Close'] = train_df['Close'].astype(str).str.replace(',', '').astype(float)
test_df['Close'] = test_df['Close'].astype(str).str.replace(',', '').astype(float)

# Normalize the data using MinMaxScaler
train_scaler = MinMaxScaler()
train_df['Normalized Close'] = train_scaler.fit_transform(train_df['Close'].values.reshape(-1, 1))

test_scaler = MinMaxScaler()
test_df['Normalized Close'] = test_scaler.fit_transform(test_df['Close'].values.reshape(-1, 1))

# Prepare data for LSTM
x_train = train_df['Normalized Close'].values[:-1].reshape(-1, 1, 1)
y_train = train_df['Normalized Close'].values[1:].reshape(-1, 1, 1)

x_test = test_df['Normalized Close'].values[:-1].reshape(-1, 1, 1)
y_test = test_df['Normalized Close'].values[1:].reshape(-1, 1, 1)

# Build the model
model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

# Evaluate the model
test_loss = model.evaluate(x_test, y_test)
print('Testing loss: ', test_loss)

# Test the model
y_pred = model.predict(x_test)

# Inverse transform the normalized values to get the actual values
y_test_actual = test_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = test_scaler.inverse_transform(y_pred.reshape(-1, 1))

# Print actual and predicted values
i = 1
print("Actual value: {:.2f}".format(y_test_actual[i][0]))
print("Predicted value: {:.2f}".format(y_pred_actual[i][0]))
