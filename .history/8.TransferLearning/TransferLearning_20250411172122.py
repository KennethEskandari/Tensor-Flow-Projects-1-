# Importing Libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# Load the Dataset
(train_ds, val_ds), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

# Constants
image_size = 160
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

# Preprocessing: Resize and Normalize
def preprocess(image, label):
    image = tf.image.resize(image, (image_size, image_size))
    image = image / 255.0
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

# Load Pretrained Model (without top)
base_model = MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add Custom Classification Head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Fine-tune the Model (optional — unfreeze and re-train)
base_model.trainable = True
