import numpy as np
import ssl
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from keras import models, layers, optimizers, losses, metrics
from sklearn.metrics import mean_absolute_error

# Allow downloading without SSL verification (for older systems)
ssl._create_default_https_context = ssl._create_unverified_context

# Load data (top 10,000 words)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Decode example review (optional, just for exploration)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print("Sample Decoded Review:\n", decoded_review[:500], "\n")


# Vectorize sequences (one-hot encoding)
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.0
	return results


xtrain = vectorize_sequences(train_data)
xtest = vectorize_sequences(test_data)

ytrain = np.asarray(train_labels).astype('float32')
ytest = np.asarray(test_labels).astype('float32')

# Create the model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
			  loss=losses.binary_crossentropy,
			  metrics=[metrics.binary_accuracy])

# Split training data into partial training and validation sets
xval = xtrain[:10000]
partial_xtrain = xtrain[10000:]
yval = ytrain[:10000]
partial_ytrain = ytrain[10000:]

# Train the model
history = model.fit(partial_xtrain, partial_ytrain,
					epochs=20,
					batch_size=512,
					validation_data=(xval, yval))

# Plot training & validation loss
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training & validation accuracy
acc_values = history.history['binary_accuracy']
val_acc_values = history.history['val_binary_accuracy']

plt.plot(epochs, acc_values, 'ro', label='Training Accuracy')
plt.plot(epochs, val_acc_values, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# (Optional) Retrain a few more epochs
model.fit(partial_xtrain, partial_ytrain,
		  epochs=3,
		  batch_size=512,
		  validation_data=(xval, yval))

# Predict on test data
result = model.predict(xtest)

# Convert probabilities to binary predictions
y_pred = np.array([1 if score > 0.5 else 0 for score in result])

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_pred, ytest)
print("\nMean Absolute Error on test data:", round(mae, 4))
