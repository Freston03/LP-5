from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

train_img = train_img / 255.0
test_img = test_img / 255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])

model.fit(train_img, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_img, test_labels)
print("accuracy of tessting: ",test_acc)

predictions = model.predict(test_img)

predicted_labels = np.argmax(predictions, axis=1)

num_rows = 5 
num_cols = 5
num_imgs = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_imgs):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.imshow(test_img[i], cmap='gray')
    plt.axis("off")

    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plt.bar(range(10), predictions[i])
    plt.xticks(range(10))
    plt.ylim([0,1])
    plt.title(f"Predicted Label: {predicted_labels[i]}")
plt.tight_layout()
plt.show()

****************************************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load the data from local .npy files
train_img = np.load('train_img.npy')
train_labels = np.load('train_labels.npy')
test_img = np.load('test_img.npy')
test_labels = np.load('test_labels.npy')

# Normalize the images (values should be between 0 and 1)
train_img = train_img / 255.0
test_img = test_img / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into 1D array
    keras.layers.Dense(128, activation='relu'),  # Dense hidden layer with 128 neurons
    keras.layers.Dense(10, activation='softmax')  # Output layer with 10 classes (softmax)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(train_img, train_labels, epochs=10)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_img, test_labels)
print("Accuracy on testing data: ", test_acc)

# Make predictions on the test set
predictions = model.predict(test_img)

# Convert probabilities to predicted labels
predicted_labels = np.argmax(predictions, axis=1)

# Plot some test images and their predictions
num_rows = 5
num_cols = 5
num_imgs = num_rows * num_cols

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_imgs):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plt.imshow(test_img[i], cmap='gray')
    plt.axis("off")

    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plt.bar(range(10), predictions[i])
    plt.xticks(range(10))
    plt.ylim([0, 1])
    plt.title(f"Predicted Label: {predicted_labels[i]}")

plt.tight_layout()
plt.show()
