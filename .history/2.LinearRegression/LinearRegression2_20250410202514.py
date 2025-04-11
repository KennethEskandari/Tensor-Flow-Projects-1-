import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#A linear regression measures the relationship between two variables

#Loading Dataset
Views = np.array([1000, 2000, 3000, 4000, 5000, 6000])
Controversy = np.array([0, 1, 0, 1, 0, 1])

#Graphing the Dataset
plt.scatter(Views, Controversy, color='blue', label='Data')
plt.xlabel('Views')
plt.ylabel('Controversy')
plt.title('Controversy vs Views Linear Regression')
plt.grid(True)
plt.legend()
plt.show()

#In order to measure the relationship between views and controversy,
#we need to create a model

#It is at this point where it would be useful to 
#view what an actual neural network looks like

#Creating the Model
model = tf.kera.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
#Dense states that every input is connected to every output 
#units is the number of predicted outputs
#input_shape is the number of inputs *Views* 

#Optimizing The Model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')


