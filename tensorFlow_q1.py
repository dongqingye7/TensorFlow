#import the TensorFlow library into the program
import tensorflow as tf
#Load and prepare the MNIST dataset. 
mnist = tf.keras.datasets.mnist
#split dataset to training and testing
(x_train, y_train),(x_test, y_test) = mnist.load_data()
#Convert the samples from integers to floating-point numbers
x_train, x_test = x_train / 255.0, x_test / 255.0

#Build the tf.keras model by stacking layers. 
model = tf.keras.models.Sequential([
  #Flattens the input. Does not affect the batch size.
  tf.keras.layers.Flatten(),
  # Adds a densely-connected layer with 512 units to the model:
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  #randomly setting a fraction rate 0.2 of input units to 0 at each update during training time
  tf.keras.layers.Dropout(0.2),
  # Add a softmax layer with 10 output units:
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
#Select an optimizer and loss function used for training:
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Trains for 5 epochs
model.fit(x_train, y_train, epochs=5)
#Evaluate the model
model.evaluate(x_test, y_test)
