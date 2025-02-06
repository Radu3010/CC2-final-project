import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers

# Setăm dimensiunea imaginilor
img_height = 64
img_width = 64
channels = 3  # RGB

output_dir = 'generated'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Încărcăm o singură imagine clar alb-negru
def load_image(path):
    img = plt.imread(path)
    if img.shape == (img_height, img_width, channels):
        return np.array([img])
    else:
        raise ValueError("Imaginea nu are dimensiunile corecte!")

# Încarcă imaginea (folosește calea corectă)
image_path = 'landscapes1/pexels-marek-piwnicki-3907296-20066246.jpg'  # Folosește calea imaginii tale
images = load_image(image_path)
images = (images - 127.5) / 127.5  # Normalizăm între -1 și 1

# Citim imaginea corect (în loc de image_path)
img = plt.imread(image_path)

# Verificăm dacă imaginea este de dimensiunea corectă
if img.shape == (img_height, img_width, channels):
    # Extragem valoarea pixelului din colțul din dreapta jos
    pixel_value = img[1, 1]  # Pixelul din colțul din dreapta jos
    print(f"Valoarea pixelului din colțul din dreapta jos: {pixel_value}")
else:
    print(f"Imaginea nu are dimensiunile corecte: {img.shape}")

# Generatorul
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 128, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, 4, strides=2, padding='same', use_bias=False, activation='tanh')  # corect
    ])
    return model

# Discriminatorul
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 4, strides=2, padding='same', input_shape=[img_height, img_width, channels]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Creăm modelele
generator = build_generator()
discriminator = build_discriminator()

# Compilăm modelele
optimizer = tf.keras.optimizers.Adam(learning_rate=0.000000000000000000000001, beta_1=0.5)
discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Modelul GAN (generator + discriminator)
z = layers.Input(shape=(100,))
generated_image = generator(z)
discriminator.trainable = False
validity = discriminator(generated_image)
gan = tf.keras.Model(z, validity)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')

# Funcție pentru antrenarea GAN-ului
def train_gan(epochs, batch_size):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # Selectăm un batch aleator de imagini reale
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]

        # Generăm imagini false
        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_images = generator.predict(noise)

        # Antrenăm discriminatorul
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Antrenăm generatorul
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Afisăm progresul
        if epoch % 50 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            save_generated_images(epoch)  # Salvăm imaginile generate la fiecare 50 de epoci

# Funcție pentru a salva imaginile generate
def save_generated_images(epoch):
    noise = np.random.normal(0, 1, (16, 100))
    generated_images = generator.predict(noise)

    # Verifică valorile minime și maxime ale imaginii generate
    print("Minim și maxim valori imagine generată:", np.min(generated_images), np.max(generated_images))

    generated_images = (generated_images + 1) / 2.0  # Inversăm normalizarea pentru a fi între [0, 1]
    generated_images = np.clip(generated_images, 0, 1)  # Asigurăm că valorile sunt între 0 și 1

    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    cnt = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig(f"generated_images_{epoch}.png")
    plt.close()

# Antrenăm GAN-ul
train_gan(epochs=5000, batch_size=16)