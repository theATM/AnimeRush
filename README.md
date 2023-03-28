# Anime Face Generation using Generative Adversarial Network
In the world of NFTs and generated collectibles, pictures of anime girls can be very attractive and lucrative digital good to posses. The recent advancements in Deep Learning and Artificial Intelligence open up new world of possibilities. The generation of such portraits can be accomplished with the easy generative adversarial network. 
This work demonstrates simple and straight forward ways of using the DC-GAN architecture 
to obtain bright new and unique ’waifus’ the heart can desire. 

This project has been tasked by the University of Oulu by the Faculty of Information Technology and Electrical Engineering as a part of the Deep Learning course. 

#### Example of the Generated Images:

![9 6](https://user-images.githubusercontent.com/48883111/228368978-2ffc3bf9-d886-4207-8647-1ba87a303841.png)


### Deep Learning Project Structure
* project_GAN.ipynb - main notebook with unimproved GAN training provided by the university staff
* hyper_GAN.ipynb - proposed improvements to the GAN training
* critic_climping_WGAN - experiment conducted on the WGAN architecture (code taken from https://github.com/Zeleni9/pytorch-wgan)
* xfunction - directory for the utility functions
  * transforms.py - file with proposed transform
  * models.py - different DCGAN architectures tested during experiments


### Data Usage:
Create Dataset directory and inside
 * images - data given by the OULU Uni
 * images2 - dataset Anime Face Dataset Database Contents License (DbCL) v1.0
  https://www.kaggle.com/datasets/splcher/animefacedataset
 * rem1 - preprocessed images from dataset https://www.kaggle.com/datasets/andy8744/rezero-rem-anime-faces-for-gan-training CC0: Public Domain
 * rem2 - fake images from rem-anime-faces dataset
 * testB - selfie2anime testing anime set 
 * trainB - selfie2anime training anime set https://www.kaggle.com/datasets/arnaud58/selfie2anime CC0: Public Domain

### Project can be found on GitHub:
https://github.com/theATM/AnimeRush

### Author:
Aleksander Madajczak
