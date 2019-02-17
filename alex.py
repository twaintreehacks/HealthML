import tensorflow as tf
import keras as k
import numpy as np
from tensorflow.keras import layers
from random import shuffle
import matplotlib.pyplot as plt

file = open("heart.csv", "r")
firstLine = True
data = []
characteristics = []

for x in file:
    if firstLine:
        firstLine = False
        characteristics = x.strip().split(",")
        characteristics[0] = 'age'
    else:
        nextData = {}
        sampleData = x.strip().split(",")
        for num in range(0, len(sampleData)):
            nextData[characteristics[num]] = sampleData[num]
        data.append(nextData)

shuffle(data)
testingSamples = data[230:]
trainingSamples = data[:230]

trainingValues = []
trainingResults = []

for elem in trainingSamples:
    allAttributes = list(elem.values())
    trainingValues.append(allAttributes[:-1])
    if allAttributes[-1] == '1':
        trainingResults.append([1, 0])
    else:
        trainingResults.append([0, 1])

testingResults = []
testingValues = []

for elem in testingSamples:
    allAttributes = list(elem.values())
    testingValues.append(allAttributes[:-1])
    if allAttributes[-1] == '1':
        testingResults.append([1, 0])
    else:
        testingResults.append([0, 1])

model = tf.keras.Sequential()
model.add(layers.Dense(64, input_dim = len(trainingValues[0]), activation = 'tanh'))
model.add(layers.Dense(2, activation = 'sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(np.array(trainingValues), np.array(trainingResults), epochs=300, batch_size=7)

plt.plot(history.history['acc'])
plt.title('Heart Disease Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy'], loc='upper left')
plt.show()
