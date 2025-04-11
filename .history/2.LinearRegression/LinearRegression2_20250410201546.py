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



