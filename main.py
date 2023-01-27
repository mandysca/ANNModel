# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn

print(tf.__version__)

# Initialize Ann
# Sequential class object - as a sequence of layers

ann = tf.keras.models.Sequential();

# Adding the input layer and the first hidden layer
# fully connected layer object of class dense
# Input neurons in units as argument

ann.add(tf.keras.layers.Dense(6))
