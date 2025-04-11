import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam

# -----------------------
# Loading Data 
# -----------------------
(x_train, _), (_, _) = mnist.load_data()

# Preprocessing Data: Normalize and reshape (scale to [0, 1])
x_train = x_train / 255.0
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))

# -----------------------
# Creating the Generator
# -----------------------
def build_generator():
    model = Sequential([
        # Use an Input layer instead of 'input_dim'
        Input(shape=(100,)),
        Dense(128, activation='relu'),
        Dense(784, activation='sigmoid'),
        Reshape((28, 28, 1))
    ])
    return model

# -----------------------
# Creating the Discriminator
# -----------------------
def build_discriminator():
    model = Sequential([
        # Use an Input layer instead of passing input_shape in Flatten
        Input(shape=(28, 28, 1)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# -----------------------
# Creating the GAN
# -----------------------
def build_gan(generator, discriminator):
    # Freeze the discriminator when training the GAN
    discriminator.trainable = False
    model = Sequential([
        generator,
        discriminator
    ])
    return model

# -----------------------
# Compiling the Models
# -----------------------
# Build and compile the discriminator first
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', 
                      optimizer=Adam(learning_rate=0.0002, beta_1=0.5), 
                      metrics=['accuracy'])

# Build the generator
generator = build_generator()

# Build and compile the GAN (generator + frozen discriminator)
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', 
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5), 
            metrics=['accuracy'])

# -----------------------
# Function to Plot Generated Images
# -----------------------
def plot_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        # Display the generated image
        plt.imshow(generated_images[i].reshape(28, 28), interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_image_epoch_{epoch}.png")
    plt.close()

# -----------------------
# Training the GAN
# -----------------------
def train_gan(epochs=100, batch_size=128):
    batch_count = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        for i in range(batch_count):
            # -----------------------
            # Train the Discriminator
            # -----------------------
            # Generate noise and produce fake images
            noise = np.random.normal(0, 1, size=(batch_size, 100))
            generated_images = generator.predict(noise)

            # Get a random batch of real images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]

            # Labels for real and fake images
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            # Set discriminator trainable for its own training
            discriminator.trainable = True
            # Train on real images and on fake images separately and average loss
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------------
            # Train the Generator (via GAN Model)
            # -----------------------
            # Freeze discriminator when training GAN
            discriminator.trainable = False
            noise = np.random.normal(0, 1, size=(batch_size, 100))
            # We want the generator to fool the discriminator --> use real_labels as target
            g_loss = gan.train_on_batch(noise, real_labels)

        # End of epoch - print progress and plot images
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]*100:.2f}%, G Loss: {g_loss[0]:.4f}")
            plot_generated_images(epoch)

# -----------------------
# Start Training and Viewing the Results
# -----------------------
train_gan(epochs=100, batch_size=128)
