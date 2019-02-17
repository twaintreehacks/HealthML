import tensorflow as tf
import keras as k
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

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

model = k.models.Sequential([
    k.layers.Dense(64, input_dim = len(trainingValues[0])),
    k.layers.Activation('tanh'),
    k.layers.Dense(2),
    k.layers.Activation('sigmoid'),
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(np.array(trainingValues), np.array(trainingResults), epochs=310, batch_size=7)
print(model.predict(np.array(testingValues)))
print(testingResults)
print(model.evaluate(np.array(testingValues), np.array(testingResults)))

print(history.history)

plt.plot(history.history['acc'])
plt.title('Heart Disease Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy'], loc='upper left')
plt.show()

corr = trainingResults.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.show()

#
# #This is Alex's enhanced machine learning ML implementation
# import tensorflow as tf
# import keras as k
# from random import shuffle
# from tensorflow.keras import layers
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# file = open("heart.csv", "r")
# first = True
# characteristics = []
# data = []
# dataResults = []
#
# for x in file:
#     if first:
#         characteristics = x.strip().split(",")
#         characteristics[0] = 'age'
#         first = False
#     else:
#         vals = x.strip().split(",")
#
#         dataPoint = {}
#         for ii in range(len(vals)):
#             dataPoint[characteristics[ii]] = vals[ii]
#         data.append(dataPoint)
# shuffle(data)
# print(data)
#
#
# trainingSamples = data[:200]
# testingSamples = data[200:]
#
# trainingVals = []
# trainingResults = []
#
#
# testVals = []
# testResults = []
#
# for d in testingSamples:
#     vals = list(d.values())
#     testVals.append(vals[:-1])
#     if vals[10] == '1':
#         testResults.append([1, 0])
#     else:
#         testResults.append([0, 1])
#     del vals[10]
#     testVals.append(vals)
#
#
# file.close()
