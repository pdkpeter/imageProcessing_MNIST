import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers #neural network API
from tensorflow.keras.datasets import mnist

"""
The training set is a subset of the data set used to train a model.

x_train is the training data set.
y_train is the set of labels to all the data in x_train.
The test set is a subset of the data set that you use to test your model after the model has gone through initial vetting by the validation set.

x_test is the test data set.
y_test is the set of labels to all the data in x_test.
The validation set is a subset of the data set (separate from the training set) that you use to adjust hyperparameters.
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data() # load dataset
#(60000, 28, 28) - 60000 images where they are all 28 by 28
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255 # flatens into one column - /255 is to normalize values into 0-1
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255

# Sequential API (1 input mapped to 1 output only)
model = keras.Sequential(
    [
        layers.Dense(512, activation='relu'), #512 nodes
        layers.Dense(256, activation='relu'),
        layers.Dense(10), #output layer
    ]
)
#Functional API (More inputs and outputs)
# inputs = keras.Input(shape=(784))
# x = layers.Dense(512, activation='relu')(inputs)
# x = layers.Dense(512, activation='relu')(x)
# outputs = layers.Dense(10, activation='softmax')(x)
model.compile( #how to configure the training part of the network
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2) #more training. receives the data
#x is input data(samples) - y is the target data(labels)

model.evaluate(x_test, y_test, batch_size=32, verbose=2)