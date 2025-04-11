from tensorflow.keras import layers, models
import tensorflow as tf
import matplotlib.pyplot as plt


#Loading Dataset
mnist = tf.keras.datasets.mnist #This is a dataset of handwritten digits

#Splitting the dataset into training and testing data
(x_train, y_train), (x_test,y_test) = mnist.load_data() 


#Processing Data 
x_train = x_train / 255.0 #Normalizing the data
x_test = x_test / 255.0

#Flatten the images
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

#Loading the model 
model = models.Sequential([
    layers.Flatten(input_shape=(28*28,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#Optmizing the Data
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

#Training the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

#Evaluating the model
print("Final Loss",history.history['loss'][-1])
print("Final Accuracy",history.history['accuracy'][-1])

#Visualizing the dataset
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.grid(True)
plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.grid(True)
plt.show()




