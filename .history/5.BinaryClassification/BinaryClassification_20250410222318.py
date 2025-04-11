import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#Loading Data 
#A big difference between this and the projects from before
# is that we are loaiding the data from a URL
#and using pandas to load the data
url = "/Users/kennetheskandari/TensorFlowProjects1/Tensor-Flow-Projects-1-/5.BinaryClassification/smsspamcollection/SMSSpamCollection"

df = pd.read_csv(url, sep="\t",names=["label", "message"])
df.head(10)

#Data processing 

#Encoding the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

#Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['message'],df['label'],test_size=0.2,random_state=42)

#Vecrtorizing the data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

print(x_train_vec.shape)

#Creating the model
model = Sequential()
model.add (Dense(128, activation='relu',input_dim=x_train_vec.shape[1]))
model.add (Dropout(0.5))
model.add (Dense(1, activation='sigmoid'))

#Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_vec, y_train, epochs=5, batch_size=32, validation_data=(x_test_vec, y_test), verbose=1)

#Evaluating the model
loss, accuracy = model.evaluate(x_test_vec, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")





