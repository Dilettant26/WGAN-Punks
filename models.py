
import tensorflow as tf
from tensorflow import keras
from preconfig import *
from functools import partial
from utils.losses import generator_loss, discriminator_loss, gradient_penalty


conv_window = 3

#===========generator Model============

def Generator():
    
    inputs = keras.Input(shape=(1, 1, z_dim))
    x = keras.layers.Dense(4 * 4 * max_filter, use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    # 1x1 -> 4x4
    x = keras.layers.Reshape((4, 4, max_filter))(x)
    x = keras.layers.Conv2DTranspose(max_filter, conv_window, strides=1, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    # 4x4 -> 8x8
    x = keras.layers.Conv2DTranspose(max_filter/2, conv_window, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    # 8x8 -> 16x16
    x = keras.layers.Conv2DTranspose(max_filter/4, conv_window, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    # 16x16 -> 32x32
    x = keras.layers.Conv2DTranspose(3, conv_window, strides=2, padding='same', use_bias=False)(x)
    outputs = keras.layers.Activation('tanh')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='Generator')


#===========Discriminator Model============

def Discriminator():
    
    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNEL))
    # 32x32 -> 16x16
    x = keras.layers.Conv2D(max_filter/16, conv_window, strides=2, padding='same', use_bias=True)(inputs)
    x = keras.layers.LeakyReLU(0.2)(x)
    # 16x16 -> 8x8
    x = keras.layers.Conv2D(max_filter/8, conv_window, strides=2, padding='same', use_bias=True)(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    # 8x8 -> 4x4
    x = keras.layers.Conv2D(max_filter/4, conv_window, strides=2, padding='same', use_bias=True)(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    # 4x4 -> 1x1
    x = keras.layers.Conv2D(max_filter/2, conv_window, strides=2, padding='same', use_bias=True)(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    #Final
    x = keras.layers.Conv2D(max_filter, conv_window, strides=1, padding='same', use_bias=True)(x)
    outputs = keras.layers.Dense(1)(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='Discriminator')

# Function for training of Generator
@tf.function
def train_generator(generator, discriminator, g_optimizer):
    with tf.GradientTape() as tape:
        # sample data
        random_vector = tf.random.normal(shape=((BATCH_SIZE, 1, 1, z_dim)))
        # create image
        fake_img = generator(random_vector, training=True)
        # predict real or fake
        fake_logit = discriminator(fake_img, training=True)
        # calculate generator loss
        g_loss = generator_loss(fake_logit)

    # Update Gradients
    gradients = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return g_loss

# Function for training of Discriminator
@tf.function
def train_discriminator(generator, discriminator, d_optimizer, real_img):
    with tf.GradientTape() as t:
        z = tf.random.normal(shape=(BATCH_SIZE, 1, 1, z_dim))
        fake_img = generator(z, training=True)

        real_logit = discriminator(real_img, training=True)
        fake_logit = discriminator(fake_img, training=True)

        real_loss, fake_loss = discriminator_loss(real_logit, fake_logit)
        gp = gradient_penalty(partial(discriminator, training=True), BATCH_SIZE ,real_img, fake_img)

        d_loss = (real_loss + fake_loss) + gp * gradient_penalty_weight

    # Update Gradients
    D_grad = t.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(D_grad, discriminator.trainable_variables))

    return real_loss + fake_loss, gp



