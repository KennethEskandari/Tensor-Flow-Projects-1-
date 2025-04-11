import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#A linear regression measures the relationship between two variables

#Loading Dataset
Views = np.array([1000, 2000, 3000, 4000, 5000, 6000])
Controversy = np.array([0, 1, 0, 1, 0, 1])

#In order to measure the relationship between views and controversy,
#we need to create a model
#It is at this point where it would be useful to 
#view what an actual neural network looks like

#Creating the Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

#Dense states that every input is connected to every output 
#units is the number of predicted outputs
#input_shape is the number of inputs *Views* 

#The next step is optmizing the model. 
#We are going to give the model some perameters to work with
#so it behaves well. 

#Optimizing The Model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')

#tf.keras.optomizers.SGD has the job of updating weights, so the model can learn better
#learning_rate is the rate at which the model learns
#loss is the function that measures how well the model is doing (we do not want this to be high)

#Training The Model
history = model.fit(Views, Controversy, epochs=1000, verabase=0)

#model.fit is basically giving the model the variables to learn from (like givin it homework)
#epochs is the number of times the model will learn from the dataset
#verbase is the amount of information we want to see while the model is learning (I like 0)

plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.grid(True)
plt.show()

#Evaluating The Model
print("Final Loss",history.history['loss'][-1])



