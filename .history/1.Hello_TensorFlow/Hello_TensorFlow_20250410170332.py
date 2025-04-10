import tensorflow as tf

#3 Main Data Types in TensorFlow
print(
    "In TensorFlow, data is divided into 3 main categories: Scalars, Vectors, and Matrices."
    "Scalars are single values, Vectors are 1D arrays, and Matrices are 2D arrays."
      )

#Scalar 
scalar = tf.constant(1)

#Vector
vector = tf.constant([1,10])


#Matrix
matrix = tf.constant([[1,10],[2,20]])


#Mathematics With Tensors
x = tf.constant(10) #Scalar
y = tf.constant([1,2,3]) #Vector
z = tf.constant([[1,2],[3,4]]) #Matrix 

Addition = tf.add(x, y) #Scalar + Vector
Subtraction = tf.subtract(z,x) #Matrix - Scalar
Multiplication = tf.multiply(y,z) #Vector * Matrix
Division = tf.divide(z,x) #Matrix / Scalar

print(Addition)
print(Subtraction)
print(Multiplication)
print(Division)

