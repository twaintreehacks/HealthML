import keras as k
import numpy as np
from random import shuffle

file = open("heart.csv", "r")
first = True
attributes = []
data = []

for line in file:
    if first:
        attributes = line.strip().split(",")
        attributes[0] = 'age'
        first = False
    else:
        vals = line.strip().split(",")
        dataPoint = {}
        for ii in range(0, len(vals)):
            dataPoint[attributes[ii]] = vals[ii]
        data.append(dataPoint)

shuffle(data)

trainingData = data[:200]
testData = data[200:]

trainingVals = []
trainingResults = []

for d in trainingData:
    vals = list(d.values())
    trainingVals.append(vals[:-1])
    if vals[-1] == '1':
        trainingResults.append([1, 0])
    else:
        trainingResults.append([0, 1])


testVals = []
testResults = []

for d in testData:
    vals = list(d.values())
    testVals.append(vals[:-1])
    if vals[-1] == '1':
        testResults.append([1, 0])
    else:
        testResults.append([0, 1])

model = k.models.Sequential([
    k.layers.Dense(64, input_dim = len(trainingVals[0])),
    k.layers.Activation('softmax'),
    k.layers.Dense(2),
    k.layers.Activation('softmax')
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(np.array(trainingVals), np.array(trainingResults), epochs=20, batch_size=8)
print(model.predict(np.array(testVals)))
print(testResults)
print(model.evaluate(np.array(testVals), np.array(testResults)))
