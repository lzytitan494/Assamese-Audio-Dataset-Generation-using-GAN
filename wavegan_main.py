import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Defining the GAN architecture
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*128, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 128)))
    assert model.output_shape == (None, 16, 16, 128) 

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Defining the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Setting up the optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Defining the training loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        print("\n\nGenerator Loss: ",gen_loss)
        print("\n\nDiscriminator Loss: ",disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Set up the data pipeline
data_dir = "C:/Users/Yash Vardhan Gautam/OneDrive - iiitnr.edu.in/Documents/Projects/MLA Project/data"
file_list = os.listdir(data_dir)
train_files = [os.path.join(data_dir, f) for f in file_list]
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [64, 64])
    return img

def load_and_preprocess_image(path):
    return preprocess_image(path)

train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
train_dataset = train_dataset.shuffle(len(train_files))
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
generator = make_generator_model()
discriminator = make_discriminator_model()

EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 100
seed = tf.random.normal([num_examples_to_generate, noise_dim])

import matplotlib.pyplot as plt
output_dir="C:/Users/Yash Vardhan Gautam/OneDrive - iiitnr.edu.in/Documents/Projects/MLA Project/generated_images"
def generate_and_save_audio(model, epoch, test_input):
    predictions = model(test_input, training=False)
    print(type(predictions))
    fig = plt.figure(figsize=[1,1])
    for i in range(predictions.shape[0]):
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"generated_{i}.png"))
    # plt.savefig(os.path.join(output_dir, f"generated_{epoch}.png"))
    plt.close(fig)

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        if epoch % 10 == 0:
            generate_and_save_audio(generator, epoch + 1, seed)    


train(train_dataset,2000)
generator.save('Audio_Genearator.h5')
discriminator.save('Audio_Discriminator.h5')
