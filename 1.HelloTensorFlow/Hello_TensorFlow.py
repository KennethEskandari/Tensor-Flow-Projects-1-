import tensorflow as tf

#3 Main Data Types in TensorFlow

#Scalar 
scalar = tf.constant(1)

#Vector
vector = tf.constant([1,10])


#Matrix
matrix = tf.constant([[1,10],[2,20]])


#Inrtoducing tf.Variables
#Variables are tensors that can be changed, unlike constants.
scalar2 = tf.variable(1)
vector2 = tf.variable([1,10])
matrix2 = tf.variable([[1,10],[2,20]])

#We change tensor variables using assign, add and sub
scalar2.assign(2)
vector2.assign([2,20])
matrix2.assign([[2,20],[3,30]])

#We can also use add and sub
scalar2.assign_add(2)
vector2.assign_sub([2,20])
matrix2.assign_add([[2,20],[3,30]]) 



#Mathematics With Tensors

#Variables
a = tf.constant(10) #Scalar
b = tf.constant([10,20]) #Vector
c = tf.constant([[10,20],[30,40]]) #Matrix

x = tf.constant(10) #Scalar
y = tf.constant([1,2,3]) #Vector
z = tf.constant([[1,2],[3,4]]) #Matrix 
