"""
Created between Oct. 2021 - Dec. 2021
@author: selienamei
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import time


AUTOTUNE = tf.data.AUTOTUNE


#building generator
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result

def Generator():
    inputs = layers.Input(shape=[256,256,3])

    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4)] # (bs, 1, 1, 512)
    
    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4)] # (bs, 128, 128, 128)
    

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


#building discriminator
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)



#building cyclegan
class CycleGan(keras.Model):
    def __init__(
        self,
        a_generator,
        b_generator,
        a_discriminator,
        b_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.a_gen = a_generator
        self.b_gen = b_generator
        self.a_disc = a_discriminator
        self.b_disc = b_discriminator
        self.lambda_cycle = lambda_cycle
        
    def compile(
        self,
        a_gen_optimizer,
        b_gen_optimizer,
        a_disc_optimizer,
        b_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.a_gen_optimizer = a_gen_optimizer
        self.b_gen_optimizer = b_gen_optimizer
        self.a_disc_optimizer = a_disc_optimizer
        self.b_disc_optimizer = b_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        
    def train_step(self, batch_data):
        real_a, real_b = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            # b to a back to b
            fake_a = self.a_gen(real_b, training=True)
            cycled_b = self.b_gen(fake_a, training=True)

            # a to b back to a
            fake_b = self.b_gen(real_a, training=True)
            cycled_a = self.a_gen(fake_b, training=True)

            # generating itself
            same_a = self.a_gen(real_a, training=True)
            same_b = self.b_gen(real_b, training=True)

            # discriminator used to check, inputing real images
            disc_real_a = self.a_disc(real_a, training=True)
            disc_real_b = self.b_disc(real_b, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_a = self.a_disc(fake_a, training=True)
            disc_fake_b = self.b_disc(fake_b, training=True)
            
            # evaluates generator loss
            a_gen_loss = self.gen_loss_fn(disc_fake_a)
            b_gen_loss = self.gen_loss_fn(disc_fake_b)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_a, cycled_a, self.lambda_cycle) + self.cycle_loss_fn(real_b, cycled_b, self.lambda_cycle)

            # evaluates total generator loss
            total_a_gen_loss = a_gen_loss + total_cycle_loss + self.identity_loss_fn(real_a, same_a, self.lambda_cycle)
            total_b_gen_loss = b_gen_loss + total_cycle_loss + self.identity_loss_fn(real_b, same_b, self.lambda_cycle)

            # evaluates discriminator loss
            a_disc_loss = self.disc_loss_fn(disc_real_a, disc_fake_a)
            b_disc_loss = self.disc_loss_fn(disc_real_b, disc_fake_b)
            
        # Calculate the gradients for generator and discriminator
        a_generator_gradients = tape.gradient(total_a_gen_loss,
                                                  self.a_gen.trainable_variables)
        b_generator_gradients = tape.gradient(total_b_gen_loss,
                                                  self.b_gen.trainable_variables)

        a_discriminator_gradients = tape.gradient(a_disc_loss,
                                                      self.a_disc.trainable_variables)
        b_discriminator_gradients = tape.gradient(b_disc_loss,
                                                      self.b_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.a_gen_optimizer.apply_gradients(zip(a_generator_gradients,
                                                 self.a_gen.trainable_variables))

        self.b_gen_optimizer.apply_gradients(zip(b_generator_gradients,
                                                 self.b_gen.trainable_variables))

        self.a_disc_optimizer.apply_gradients(zip(a_discriminator_gradients,
                                                  self.a_disc.trainable_variables))

        self.b_disc_optimizer.apply_gradients(zip(b_discriminator_gradients,
                                                  self.b_disc.trainable_variables))

        return {
            "a_gen_loss": total_a_gen_loss,
            "b_gen_loss": total_b_gen_loss,
            "a_disc_loss": a_disc_loss,
            "b_disc_loss": b_disc_loss
        }


#loss function
def discriminator_loss(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1

def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss






