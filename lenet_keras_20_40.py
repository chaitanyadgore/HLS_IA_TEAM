#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 2019
@author: S. Bilavarn
"""

# Imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils 
from keras.datasets import mnist

# load data
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()

trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype('float32')
testData = testData.astype('float32')

trainData = trainData / 255.0
testData = testData / 255.0

trainLabels = np_utils.to_categorical(trainLabels)
testLabels = np_utils.to_categorical(testLabels)
num_classes = testLabels.shape[1]
print(trainData.shape)

# LeNet model
model = Sequential()
model.add(Conv2D(20,(5,5), input_shape = (28,28,1), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(40,(5,5), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(400,activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

model.fit(trainData, trainLabels, batch_size=128, epochs=20, verbose=1)

#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('lenet_weights.hdf5')    

scores = model.evaluate(testData,testLabels,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
