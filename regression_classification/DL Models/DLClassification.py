# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score


#Define data output columns for each cluster, and different sets of Input data
ClusterA=[0,1,2]
ClusterB=[3,4,5]
ClusterC=[6,7,8]
ClusterD=[9,10,11]
ClusterE=[12,13,14]
ClusterF=[15,16,17]
alldata=[0,1,2,3,4,5,6,7,8,9,10,11]
time=[0,1,2,3,4,5,8,11]
popularity=[0,1,2,3,4,5,6,7,8,11]
weather=[0,1,2,3,4,5,8,9,10,11]

###Data Import###
datasetX2 = numpy.loadtxt("Datasets/x_2018.csv", delimiter=",", skiprows=1, usecols=weather)
datasetY2 = numpy.loadtxt("Datasets/y_2018Class.csv", delimiter=",", skiprows=1, usecols=ClusterA)
datasetX = numpy.loadtxt("Datasets/x_2017.csv", delimiter=",", skiprows=1, usecols=weather)
datasetY = numpy.loadtxt("Datasets/y_2017Class.csv", delimiter=",", skiprows=1, usecols=ClusterA)

###test size 0.25 for 2017, 0 for 2018###
X_train, X_test, Y_train, Y_test = train_test_split(datasetX, datasetY, test_size=0.25, random_state=42)

###Comment the following for 2017, uncomment for 2018###
#X_test=datasetX2
#Y_test=datasetY2



model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=10, activation=tf.nn.relu),  
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),   
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='linear'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy', 'accuracy'])

model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=0)
Predictions = model.predict(X_test)
b = numpy.zeros_like(Predictions)
b[numpy.arange(len(Predictions)), Predictions.argmax(1)] = 1
Predictions = b
numpy.savetxt("actualY.csv", Y_test, delimiter=",", fmt='%i')
numpy.savetxt("errors.csv", Predictions-Y_test, delimiter=",", fmt='%i')
model.evaluate(X_test, Y_test)