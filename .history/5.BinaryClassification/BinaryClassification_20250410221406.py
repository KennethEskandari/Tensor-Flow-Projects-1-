import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

#Loading Data 
#A big difference between this and the projects from before
# is that we are loaiding the data from a URL
#and using pandas to load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

df = pd.read_csv(url, sep="\t",names=["label", "message"])
df.head(10)

#Data processing 

#Encoding the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

#Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['messege'],df['label'],test_size=0.2,random_state=42)

#Vecrtorizing the data
vectorizer = TfidVectorizer(stop_words='english', max_features=5000)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

print(x_train_vec.shape)




