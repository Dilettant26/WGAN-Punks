
import datetime
import tensorflow as tf

# input dimension for Generator
z_dim = 128

# Attributes of Input Images
HEIGHT, WIDTH, CHANNEL = 32, 32, 3

# Batch Size
BATCH_SIZE = 64

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Critic updates per generator update
n_iter_critic = 5

#Learning Rates
learnRateD = 0.0001
learnRateG = 0.0001

# Gradient penalty weight
gradient_penalty_weight = 10.0

#Filters for Models
max_filter = 256

