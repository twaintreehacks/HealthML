import keras as k
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







