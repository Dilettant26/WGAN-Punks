
import os
import tensorflow as tf
import models as models
import train 
import argparse
from helper import *
from preconfig import *
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

def main():

    # Parse Arguments from console
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default='punks') # Argument for directory
    parser.add_argument("--epochs", type=int, default=100) # Argument for Number of epochs
    parser.add_argument("--check_epochs", type=int, default=100) # Argument for Number of epochs

    args = parser.parse_args()
    type = args.type
    epochs = args.epochs
    checkepochs = args.check_epochs

    #Set directory for training checkpoints
    checkpoint_dir = './model/' + type
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # Get current epoch from /model/punks/Current_Epoch.txt
    h = open('model/' + type + '/Curr_Epoch.txt', 'r')
    content = h.readlines()
    lastepoch = int(content[0])

    # Build models
    generator = models.Generator()
    discriminator = models.Discriminator()

    # Build Training Dataset
    list_ds = tf.data.Dataset.list_files('./Input_images/' +  type + '/*.png', shuffle=False)
    train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_dataset = configure_for_performance(train_ds, False)

    # Define Optimizers
    g_optimizer = tf.keras.optimizers.Adam(learnRateG, beta_1=0.5, beta_2=0.9)
    d_optimizer = tf.keras.optimizers.Adam(learnRateD, beta_1=0.5, beta_2=0.9)

    #Define Training Checkpoint
    checkpoint = tf.train.Checkpoint(generator_optimizer = g_optimizer,
                                    discriminator_optimizer = d_optimizer,
                                    generator = generator,
                                    discriminator = discriminator)

    #Load Checkpoint if current epoch > 0
    if lastepoch > 0:

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Train actual Model
    train.train(train_dataset, 
            epochs, 
            lastepoch,
            checkepochs,
            type, 
            generator, 
            discriminator,
            g_optimizer, 
            d_optimizer,
            checkpoint,
            checkpoint_prefix)

    #Generate Final images
    train.generate_final(generator, 100)



if __name__ == '__main__':

    main()

