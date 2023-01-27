# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn

print(tf.__version__)


# 1. --------- Data preprocessing --------------
# Read the data set from file

dataset = pd.read_csv('/Users/mandeepsingh/Pycharmprojects/pythonproject/Churn_Modelling.csv')

# Get data set into the variables
# Variables
x = dataset.iloc[:, 3:-1].values
# Outcomes
y = dataset.iloc[:, -1].values

print(x)
print(y)

# Label encoding of the Gender column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Label encoder automatically assigns label
x[:, 2] = le.fit_transform(x[:, 2])

# one hot encoding for Country values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder= 'passthrough')
x = np.array(ct.fit_transform(x))

print("transformed data:")
print(x)

# Split data set into a training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print("Training sets")
print(x_train)
print(x_test)

# 2. --------- Initialize ANN ---------------------

# Initialize Ann
# Sequential class object - as a sequence of layers

ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# fully connected layer object of class dense
# Input neurons in units as argument as number of neurons in the layer
# and activation argument

ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

# Add a second hidden layer

ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

# add an output layer, that has one neuron

ann.add(tf.keras.layers.Dense(units = 1, activation= 'sigmoid'))

# --------- Training ANN ---------------------

# Compile
ann.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# train
# Classic value typically choosen in 32
#ann.fit(x_train, y_train, batch_size= 32, epochs= 100)
