#The First Real Model 

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

#Loading Dataset
square_footage = np.array([1500, 1600, 1700, 1800, 1900, 2000])
prices = np.array([300000, 320000, 340000, 360000, 380000, 400000])

#Visualizing the Dataset
plt.scatter(square_footage,prices,color='blue',label='Data')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Price vs SQFT Linear Regression')
plt.grid(True)
plt.legend()
plt.show()

