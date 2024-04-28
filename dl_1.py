import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('HousingData.csv')

# Prepare the data
x = df.drop("MEDV", axis=1).values
y = df["MEDV"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Data Preprocessing
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Building our Model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))  # Fixed input shape
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Training our Model
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_data=(x_test, y_test))

# Testing our Model
test_input = np.array([[-0.42101827, -0.50156705, -1.13081973, -0.25683275, -0.55572682,
                        0.19758953,  0.20684755, -0.34272202, -0.87422469, -0.84336666,
                        -0.32505625,  0.41244772, -0.63500406]])  # Converted to numpy array
print("Actual Output :", y_test[8])
print("Predicted Output :", model.predict(test_input))

# Evaluating our Model
mse_nn, mae_nn = model.evaluate(x_test, y_test)
print('Mean squared error on test data :', mse_nn)
print('Mean absolute error on test data :', mae_nn)

# Reshape predictions to match y_test shape
y_dl = np.reshape(y_dl, y_test.shape)

# Replace NaN values with 0
y_dl[np.isnan(y_dl)] = 0

# Calculate R2 score
r2 = r2_score(y_test, y_dl)

# Print R2 score
print('R2 Score :', r2)