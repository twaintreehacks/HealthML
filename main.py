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
    trainingResults.append(vals[-1])

testVals = []
testResults = []

for d in testData:
    vals = list(d.values())
    testVals.append(vals[:-1])
    testResults.append(vals[-1])

model = k.models.Sequential([
    k.layers.Dense(64, input_dim = len(trainingVals[0])),
    k.layers.Dense(1)
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(np.array(trainingVals), np.array(trainingResults), epochs=10, batch_size=32)

print(model.predict(np.array(testVals)))
print(testResults)
