import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

# Load training data
# Load the CSV file
df1 = pd.read_csv('/content/fashion-mnist_train.csv')

# Print the number of fields in each line
for i, row in df1.iterrows():
    print(f"Line {i+1}: {len(row)} fields")

# Check for lines with a different number of fields
if df1.shape[1] != 785:
    print("Error: Found lines with a different number of fields.")
    
x_train = df1.drop("label", axis=1).values
y_train = df1["label"].values

# Load testing data
df2 = pd.read_csv(r'/content/fashion-mnist_test.csv')
x_test = df2.drop("label", axis=1).values
y_test = df2["label"].values

# Reshape and normalize the data
x_train = x_train.reshape(60000, 28, 28, 1) / 255
x_test = x_test.reshape(10000, 28, 28, 1) / 255

# Build the model
model=Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=3, verbose=1, validation_data=(x_test,y_test))

# Test the model
predictions = model.predict(x_test)

index = 10
print(predictions[index])
final_value = np.argmax(predictions[index])
print("Actual label :", y_test[index])
print("Predicted label :", final_value)
print("Class :", class_names[final_value])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss :", loss)
print("Accuracy (Test Data) :", accuracy * 100)
