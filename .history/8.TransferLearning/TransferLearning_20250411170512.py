#Importing Libraries
import tensorflow as tf
from tensorflow.keras.processing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os 

train_dir = "data/train"
val_dir = "data/val"

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


