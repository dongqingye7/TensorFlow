# -*- coding: utf-8 -*-
"""TensorFlow_q3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sK98hvjDtE3HS0HRdU7MPSBWRLBp7UsL
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #tf.keras.layers.Dense(512, activation=tf.nn.relu),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

weights=model.layers[1].get_weights()[0]
weights=np.array(weights).T
for i in range (0,10):
  w=weights[i,:]
  w=np.reshape(w,(28,28))
  plt.imshow(w)
  plt.show()