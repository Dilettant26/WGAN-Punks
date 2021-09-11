# WGAN-Punks
This is a implementation of a Wasserstein GAN with applied gradient penalty for generating new Cryptopunks. 

The original Cryptopunks were designed by [LarvaLabs](https://www.larvalabs.com/cryptopunks).

## Introduction
I used the project [TF2-WGAN](https://github.com/KUASWoodyLIN/TF2-WGAN) from [KUASWoodyLIN](https://github.com/KUASWoodyLIN) as base and adapted it for my personal needs. 

This implies a different structure of the Generator and the Discriminator Networks. The Training of the WGAN is stable and results in good new images

## Model
The WGAN-model was trained on 10000 images of CryptoPunks. 

The structure of the Generator:

```python
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
```

The structure of the Discriminator:

```python
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
```
## Results
After 5000 Epochs of Training

<img src="./newImages/epoch5000.png" width="50%" height="50%"/>


## Usage
1. Download the CryptoPunk images into the folder ./Input_images/punks

2. Execute 
```bash
    python .\main.py --epochs 5000 --check_epochs 100
```

## Additional
You can use the model not only for CryptoPunks but for every kind of new images. You just have to create a new folder in the following directories:

 - ./Input_images/yourNewFolder  # Has to contain the input images for the WGAN
 - ./model/yourNewFolder  # Has to contain the file Curr_Epoch.txt with value 0
 - ./newImages/yourNewFolder

 The folder can have any name as long as you adapt your bash command like this:

```bash
    python .\main.py --type yourNewFolder --epochs 5000 --check_epochs 100
```

## References
1. [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
2. [TF2-WGAN](https://github.com/KUASWoodyLIN/TF2-WGAN)
3. [GAN — Wasserstein GAN & WGAN-GP](https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)
4. [Train Wasserstein GAN with Gradient Penalty](https://de.mathworks.com/help/deeplearning/ug/trainwasserstein-gan-with-gradient-penalty-wgan-gp.html)