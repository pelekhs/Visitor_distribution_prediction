# -*- coding: utf-8 -*-


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.optimizers import adam
import numpy

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(13,input_dim=13, activation='linear'),
    tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(72, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='linear')
  ])                                      
model.compile(loss='mse', optimizer='adam', metrics=['mae'])	

seed = 7
numpy.random.seed(seed)
#Define different sets of Input data
alldata=[0,1,2,3,4,5,6,7,8,9,10,11]
time=[0,1,2,3,4,5,8,11]
popularity=[0,1,2,3,4,5,6,7,8,11]
weather=[0,1,2,3,4,5,8,9,10,11]

###Data Import###
datasetX = numpy.loadtxt("Datasets/x_2017.csv", delimiter=",", skiprows=1, usecols=alldata)
datasetY = numpy.loadtxt("Datasets/y_2017.csv", delimiter=",", skiprows=1)
datasetX2 = numpy.loadtxt("Datasets/x_2018.csv", delimiter=",", skiprows=1, usecols=alldata)
datasetY2 = numpy.loadtxt("Datasets/y_2018.csv", delimiter=",", skiprows=1)

###test size 0.25 for 2017, 0 for 2018###
X_train, X_test, Y_train, Y_test = train_test_split(datasetX, datasetY, test_size=0, random_state=42)

###Comment the following for 2017, uncomment for 2018###
X_test=datasetX2
Y_test=datasetY2

model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=0)
Predictions = model.predict(X_test)
Pred = numpy.rint(Predictions)
model.evaluate(X_test, Y_test)
numpy.savetxt("actualY.csv", Y_test, delimiter=",", fmt="%i")
numpy.savetxt("error.csv", Pred-Y_test, delimiter=",", fmt="%i")