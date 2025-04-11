import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

#Loading Data 
(x_train, _), (_, _) = mnist.load_data()

#Preprocessing Data
x_train = x_train / 255.0
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))

#Creating the Generator
def build_generator():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=100))
    model.add(Dense(784, activation='sigmoid'))
    model.add(Reshape((28, 28, 1)))

    return model

#Creating the Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))  # Fixed Flatten instantiation
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

#Creating the GAN
def build_gan(generator, discriminator):  # Fixed typo in parameter name
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    return model

#Compiling the Models
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])  # Fixed argument name

generator = build_generator()

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

#Training the GAN ( I have no idea what is happening here)
def train_gan(epochs, batch_size = 128):
    batch_count = x_train.shape[0] // batch_size
    for epoch in range(epochs):
        for i in range(batch_count):
            noise = np.random.normal(0, 1, size=(batch_size, 100))
            generated_images = generator.predict(noise)

            #Images from Dataset 
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]

            #Labels for Discrimiinator
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            #Training the Discriminator
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = discriminator.train.on_batchh(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, size=(batch_size, 100))
            g_loss = gan.train_on_batch(noise, real_labels)

            if epoch % 10 == 0 :
                print(f"Epoch: {epoch}, Batch: {i}, D Loss: {d_loss[0]}, G Loss: {g_loss[0]}")
                plot_generated_images(epoch)

#Plotting the Generated Images
def plot_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_image_epoch_{epoch}.png")
    plt.close()