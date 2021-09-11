# WGAN-Punks
This is a implementation of a Wasserstein GAN with applied gradient penalty. 

## Introduction
I used the project [TF2-WGAN](https://github.com/KUASWoodyLIN/TF2-WGAN) from [KUASWoodyLIN](https://github.com/KUASWoodyLIN) as base and adapted it for my personal needs. This results in a WGAN which creates New CryptoPunks.

## Model
The WGAN-model was trained on 10000 images of [CryptoPunks](https://www.larvalabs.com/cryptopunks). 

## Usage
1. Download the CryptoPunk images into the folder ./Input_images/punks

2. Execute 
```bash
    python .\main.py --epochs 5000 --check_epochs 100
```

## Results

After 5000 Epochs of Training

<img src="./newImages/epoch5000.png" width="50%" height="50%"/>

## Additional

You can use the model not only for CryptoPunks but for every kind of new images. You just have to create a new folder in the following directories:

 - ./Input_images/
 - ./model/ # Has to contain the file Current_Epoch.txt with value 0
 - ./newImages/

 The folder can have any name you want as long as you adapt your bash command like this:

```bash
    python .\main.py --type yournewfoldername --epochs 5000 --check_epochs 100
```

## References
1. [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
2. [TF2-WGAN](https://github.com/KUASWoodyLIN/TF2-WGAN)