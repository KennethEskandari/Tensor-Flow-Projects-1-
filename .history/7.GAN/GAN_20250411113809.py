import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

#Loading Data 
(x_train, _), (_, _) = mnist.load_data()

#Preprocessing Data
x_train = x_train / 255.0
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))

#Creating the Generator
def generator():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=100))
    model.add(Dense(784, activation='sigmoid'))
    model.add(Reshape((28, 28, 1)))

    return model

#Creating the Discriminator
def discriminator():
    model = Sequential()
    model.add(Flatten, input_shape=(28, 28, 1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

#Creating the GAN
def gan(generator, discriminatotr):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    return model


