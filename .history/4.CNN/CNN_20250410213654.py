import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Loading Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Define Classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Preprocessing the Data
#Normalizing the images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#Reshaping
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
print("Training set shape:", train_images.shape)
print("Test set shape:", test_images.shape)

