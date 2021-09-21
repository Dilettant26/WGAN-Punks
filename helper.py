
from __future__ import division
from tensorflow import keras 
from preconfig import *
import tensorflow as tf

def decode_img(img):
        
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [HEIGHT, WIDTH], method='nearest')
    img = tf.cast(img, tf.float32)
    
    # resize the image to the desired size
    return img

def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.cast(img, tf.float32)
    img = (img-127.5)/127.5

    return img

def configure_for_performance(ds, augment):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=10000)

    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    if(augment):
        ds = ds.map(lambda x: augmentFunc(x), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

#------Augmentation if Necessary-----

augmentFunc = keras.Sequential([
	keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    keras.layers.experimental.preprocessing.RandomContrast([0.8, 1.2]),
    keras.layers.experimental.preprocessing.RandomZoom((-0.05, 0.05), (-0.05, 0.05)),
    keras.layers.experimental.preprocessing.RandomTranslation((-0.1, 0.1), (-0.1, 0.1))
])



