import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Loading Dataset
fashion_mnist = tf.keras.datasets.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

