Deep Learning Project
* project_GAN.ipynb - main notebook with unimproved GAN training
* hyper_GAN.ipynb - proposed improvements to the GAN training
* critic_climping_WGAN - experiment conducted on the WGAN architecture (code taken from https://github.com/Zeleni9/pytorch-wgan)
* xfunction - directory for the utility functions
  * transforms.py - file with proposed transform
  * models.py - different DCGAN architectures tested during experiments


Data Usage:
Create Dataset directory and inside
 * images - data given by the OULU Uni
 * images2 - dataset Anime Face Dataset Database Contents License (DbCL) v1.0
  https://www.kaggle.com/datasets/splcher/animefacedataset
 * rem1 - preprocessed images from dataset https://www.kaggle.com/datasets/andy8744/rezero-rem-anime-faces-for-gan-training CC0: Public Domain
 * rem2 - fake images from rem-anime-faces dataset
 * testB - selfie2anime testing anime set 
 * trainB - selfie2anime training anime set https://www.kaggle.com/datasets/arnaud58/selfie2anime CC0: Public Domain
 