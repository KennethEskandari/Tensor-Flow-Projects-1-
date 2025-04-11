#Importing Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os 

#Loading the Dataset
train_dir = "data/train"
val_dir = "data/val"

#Processing Dataset
image_size = 160
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255,)
val_datagen = ImageDataGenerator(recale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary',
    )

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary',
    )

#Loading The Model 
base_model = MobileNetV2(input_shape= (image_size, 3),include_top=False, weights='imagenet')
base_model.trainable = False

#Add Custom Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

#Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Train The Model 
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
)

base_model.trainable = True
