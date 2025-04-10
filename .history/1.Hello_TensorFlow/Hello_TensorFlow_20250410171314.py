import tensorflow as tf

#3 Main Data Types in TensorFlow

#Scalar 
scalar = tf.constant(1)

#Vector
vector = tf.constant([1,10])


#Matrix
matrix = tf.constant([[1,10],[2,20]])


#Mathematics With Tensors

#Variables
a = tf.constant(10) #Scalar
b = tf.constant([10,20]) #Vector
c = tf.constant([[10,20],[30,40]]) #Matrix

x = tf.constant(10) #Scalar
y = tf.constant([1,2,3]) #Vector
z = tf.constant([[1,2],[3,4]]) #Matrix 

#Inrtoducing tf.Variables
scalar = tf.Variable(1)
vector = tf.variable([1,10])
matrix = tf.Variable([[1,10],[2,20]])

