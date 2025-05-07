import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn import preprocessing

(X_train, Y_train), (X_test, Y_test) = keras.datasets.boston_housing.load_data()

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Train output data shape:", Y_train.shape)
print("Actual Test output data shape:", Y_test.shape)

##Normalize the data

X_train=preprocessing.normalize(X_train)
X_test=preprocessing.normalize(X_test)

#Model Building

X_train[0].shape
model = Sequential()
model.add(Dense(128,activation='relu',input_shape= X_train[0].shape))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse',optimizer='rmsprop',metrics=['mae'])

history = model.fit(X_train,Y_train,epochs=100,batch_size=1,verbose=1,validation_data=(X_test,Y_test))

results = model.evaluate(X_test, Y_test)
print(results)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show()
# Step 7: Evaluate the model
# Evaluate the model on the testing set
loss, mae = model.evaluate(X_test, Y_test)
# Print the mean absolute error
print('Mean Absolute Error:', mae)

*************************************************************
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from your local file
data = pd.read_csv("F:\Datasets\boston_price_prediction.csv")

# Separate features and target
X = data.drop('MEDV', axis=1).values
Y = data['MEDV'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Now continue the same as before (scaling + model building)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

# Train model
history = model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_data=(X_test, Y_test))

# Evaluate model
loss, mae = model.evaluate(X_test, Y_test)
print('Test Loss:', loss)
print('Mean Absolute Error:', mae)

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show()
