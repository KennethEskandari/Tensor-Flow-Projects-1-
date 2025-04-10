import tensorflow as tf

#Explanation
print("In TensorFlow, data is divided into 3 main categories: Scalars, Vectors, and Matrices."
      " Scalars are single values, Vectors are 1D arrays, and Matrices are 2D arrays.")

#Scalar 
scalar = tf.constant(1)
print("Scalar: ", scalar)

#Vector
vector = tf.constant([1,10])
print("Vector:", vector)

#Matrix
matrix = tf.constant([1,10][2,20])
print("Matrix:",matrix)

