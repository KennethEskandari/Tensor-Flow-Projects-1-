import tensorflow as tf
import matplotlib.pyplot as plt

#Loading Dataset
mnist = tf.keras.datasets.mnist #This is a dataset of handwritten digits

#Splitting the dataset into training and testing data
(x_train, y_train), (x_test,y_test) = mnist.load_data() 

#Visualizing Sample Data 
plt.figure(figsize=(10,2))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')
plt.show()



