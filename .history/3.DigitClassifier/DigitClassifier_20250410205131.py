import tensorflow as tf
import matplotlib.pyplot as plt

#Loading Dataset
mnist = tf.keras.datasets.mnist #This is a dataset of handwritten digits

#Splitting the dataset into training and testing data
(x_train, y_train), (x_test,y_test) = mnist.load_data()

