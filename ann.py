# -*- coding: utf-8 -*-
import pandas as pd
dataset = pd.read_csv('train.csv');
datasetTest = pd.read_csv('test.csv')

trainData = [dataset.iloc[:,2], dataset.iloc[:,4:8], dataset.iloc[:,11]]
trainData = pd.concat(trainData, axis=1)

testData = [datasetTest.iloc[:,1],datasetTest.iloc[:,3:7],datasetTest.iloc[:,10]]
testData = pd.concat(testData,axis=1)

outputTrain = dataset.iloc[:,1]

trainData.iloc[:,5] = trainData.iloc[:,5].fillna('N')
trainData = trainData[trainData.Embarked != 'N']

trainData.iloc[:,2] = trainData.iloc[:,2].fillna(0)
testData.iloc[:,2] = testData.iloc[:,2].fillna(0)

outputTrain.pop(61)
outputTrain.pop(829)


from sklearn.preprocessing import LabelEncoder
label_en = LabelEncoder()
trainData.iloc[:,1] = label_en.fit_transform(trainData.iloc[:,1])
trainData.iloc[:,5] = label_en.fit_transform(trainData.iloc[:,5])
testData.iloc[:,1] = label_en.fit_transform(testData.iloc[:,1])
testData.iloc[:,5] = label_en.fit_transform(testData.iloc[:,5])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
trainData = sc.fit_transform(trainData)
testData = sc.fit_transform(testData)

from keras.models import Sequential
from keras.layers import Dense

myANN = Sequential();

myANN.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu', input_dim = 6 ))
myANN.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu' ))
myANN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
myANN.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

myANN.fit(trainData, outputTrain, batch_size=7, epochs=100)



