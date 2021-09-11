
from models import *
from helper import *
from preconfig import *

#Train function
def train(dataset, epochs, LastEpoch, checkpoint_epochs, type, generator, discriminator, g_optimizer, d_optimizer, checkpoint, checkpoint_prefix):
    
    # Number of images to generate every 100 epochs
    num_examples_to_generate = 64

    #Start training loop
    for epoch in range(LastEpoch, LastEpoch + epochs):
        
        #real_img = Batches
        for real_img in dataset:

            #Train Discriminator with Batch
            d_loss, gp = train_discriminator(generator, discriminator, d_optimizer, real_img)

            #Train Generator for every n_iter_critic epochs of discriminator
            if d_optimizer.iterations.numpy() % n_iter_critic == 0:
                
                g_loss = train_generator(generator, discriminator, g_optimizer)

                print('G Loss: {:.2f}\t \tD loss: {:.2f} \t \t GP Loss {:.2f}'.format(g_loss, d_loss, gp))

    # Save the model and generate test images every 100 epochs

        if (epoch + 1) % checkpoint_epochs == 0:

            checkpoint.save(file_prefix = checkpoint_prefix)

            with open('model/' + type + '/Curr_Epoch.txt', 'w') as f:
                f.write('%d' % (epoch+1))

            sample_random_vector = tf.random.normal((num_examples_to_generate , 1, 1, z_dim))
            generate_and_save_images(generator, epoch, sample_random_vector, type)
      
        print ('Epoch {} '.format(epoch + 1))

# Function for image creation from prediction of model
def generate_and_save_images(model, epoch, test_input, type):

    predictions = model(test_input, training=False)
    predictions = tf.math.round(predictions * 255)
    predictions = tf.image.convert_image_dtype(predictions, dtype=tf.float64)

    save_images(predictions.numpy(), [8,8] ,'newImages/' + type + '/epoch' + str(epoch+1) + '.png')

# Function for image creation after training loop has ended
def generate_final( generator, num_examples_to_generate):
    
    sample_random_vector = tf.random.normal((num_examples_to_generate , 1, 1, z_dim))

    predictions = generator(sample_random_vector, training=False)
    predictions = tf.math.round(predictions * 255)

    predictions = tf.image.convert_image_dtype(predictions, dtype=tf.float64)

    for idx,image in enumerate(predictions):

        imageio.imwrite("Final_Images/Image_" + str(idx) + ".png", image.numpy())
