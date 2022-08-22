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
import cyclegan as cgan
import time

#loading images
def load(image_file):
  # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.8)
    image = tf.image.resize(image, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = (image / 127.5) - 1

    return image


#CHANGE THE PATH TO THE FOLDER WHERE IMAGES ARE STORED.
sa_data = tf.data.Dataset.list_files(str('PATH/*.jpg'))   #change the path 
sa_data = sa_data.map(load,num_parallel_calls=tf.data.AUTOTUNE)
sa_data = sa_data.batch(1) #BATCH_SIZE = 1 for better result in Unet
sa_sample = next(iter(sa_data))

luca_data = tf.data.Dataset.list_files(str('PATH/*.jpg'))    #change the path 
luca_data = luca_data.map(load,num_parallel_calls=tf.data.AUTOTUNE)
luca_data = luca_data.batch(1) #BATCH_SIZE = 1 for better result in Unet
luca_sample = next(iter(luca_data))

a_generator = cgan.Generator() # transforms images
b_generator = cgan.Generator() # transforms images

a_discriminator = cgan.Discriminator() # differentiates real images and generated images
b_discriminator = cgan.Discriminator() # differentiates real images and generated images

#training cycleGAN
a_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
b_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

a_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
b_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

   
cycle_gan_model = cgan.CycleGan(a_generator, b_generator, a_discriminator, b_discriminator)

cycle_gan_model.compile(a_gen_optimizer = a_generator_optimizer,
                        b_gen_optimizer = b_generator_optimizer,
                        a_disc_optimizer = a_discriminator_optimizer,
                        b_disc_optimizer = b_discriminator_optimizer,
                        gen_loss_fn = cgan.generator_loss,
                        disc_loss_fn = cgan.discriminator_loss,
                        cycle_loss_fn = cgan.calc_cycle_loss,
                        identity_loss_fn = cgan.identity_loss)

start = time.time()
history = cycle_gan_model.fit(tf.data.Dataset.zip((luca_data, sa_data)),epochs=10)
end = time.time()
time = end - start 
print('Training Time: ', time)


#generate luca to spirited away style
l = iter(luca_data)
for n_sample in range(10):
        example_sample = next(l)
        generated_sample = b_generator(example_sample)
        
        f = plt.figure(figsize=(15, 15))
        
        plt.subplot(121)
        plt.title('Input image')
        plt.imshow(example_sample[0] * 0.5 + 0.5)
        plt.axis('off')
        
        plt.subplot(122)
        plt.title('Generated image')
        plt.imshow(generated_sample[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()

#generate spirited away to luca style 
sa = iter(sa_data)
for n_sample in range(10):
        example_sample = next(sa)
        generated_sample = a_generator(example_sample)
        
        f = plt.figure(figsize=(15, 15))
        
        plt.subplot(121)
        plt.title('Input image')
        plt.imshow(example_sample[0] * 0.5 + 0.5)
        plt.axis('off')
        
        plt.subplot(122)
        plt.title('Generated image')
        plt.imshow(generated_sample[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()
