#The First Real Model 

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Loading Dataset
square_footage = np.array([1500, 1600, 1700, 1800, 1900, 2000])
prices = np.array([300000, 320000, 340000, 360000, 380000, 400000])

#Creating the Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 1, input_shape=[1])
])

#Optimizing The Model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')

#Training The Model
history = model.fit(square_footage, prices, epochs=500, verbose=1)

#Evaluating The Model 
print("Final Loss",history.history['loss'][-1])
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.grid(True)
plt.show()

#Visualizing the Dataset
plt.scatter(square_footage,prices,color='blue',label='Data')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Price vs SQFT Linear Regression')
plt.grid(True)
plt.legend()
plt.show()

