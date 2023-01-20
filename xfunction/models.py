from torch import nn
'''
Here are some used in the experiments architectures:
'''

class DiscriminatorExtend(nn.Module):
    def __init__(self,inchannels):
        super(DiscriminatorExtend,self).__init__()
        """
        Initialize the Discriminator Module
        :param inchannels: The depth of the first convolutional layer
        """
        self.cnn1 = nn.Sequential(
            # in: inchannels (3) x 64 x 64
            nn.Conv2d(in_channels=inchannels, out_channels=64, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # out: 64 x 32 x 32
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # out: 128 x 16 x 16
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # out: 256 x 8 x 8
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # out: 512 x 4 x 4
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # out: 1024 x 2 x 2
        )
        self.cnn6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4,stride=2, padding=1),
            nn.Flatten(),
            nn.Sigmoid(),
            # out: 1 x 1 x 1
        )

    def forward(self,x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: Discriminator logits; the output of the neural network
        """
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        return x




class DiscriminatorBuff(nn.Module):
    def __init__(self,inchannels):
        super(DiscriminatorBuff,self).__init__()
        """
        Initialize the Discriminator Module
        :param inchannels: The depth of the first convolutional layer
        """
        self.cnn1 = nn.Sequential(
            # in: inchannels(3) x 64 x 64
            nn.Conv2d(in_channels=inchannels, out_channels=256, kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4,stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
            # out: 1 x 1 x 1
        )

    def forward(self,x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: Discriminator logits; the output of the neural network
        """
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        return x



class GeneratorExtend(nn.Module):
    def __init__(self,latent_size):
        super(GeneratorExtend,self).__init__()
        """
        Initialize the Generator Module
        :param latent_size: The length of the input latent vector
        """
        self.cnnT1 = nn.Sequential(
            # in 128 x 1 x 1
            nn.ConvTranspose2d(in_channels=latent_size,out_channels=2048,kernel_size=4,stride=1,padding=0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            # out: 2048 x 4 x 4
        )
        self.cnnT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048,out_channels=512,kernel_size=4,stride=2,padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # out: 512 x 8 x 8
        )
        self.cnnT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # out: 256 x 16 x 16
        )
        self.cnnT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # out: 128 x 32 x 32
        )
        self.cnnT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=3,kernel_size=4,stride=2,padding=1, bias=False),
            nn.Tanh(),
            # out: 3 x 64 x 64
        )
    def forward(self,x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: A 3x64x64 Tensor image as output
        """
        x = self.cnnT1(x)
        x = self.cnnT2(x)
        x = self.cnnT3(x)
        x = self.cnnT4(x)
        x = self.cnnT5(x)
        return x



class GeneratorBuff(nn.Module):
    def __init__(self,latent_size):
        super(GeneratorBuff,self).__init__()
        """
        Initialize the Generator Module
        :param latent_size: The length of the input latent vector
        """
        self.cnnT1 = nn.Sequential(
            # in: latent_size (128) x 1 x 1
            nn.ConvTranspose2d(in_channels=latent_size,out_channels=4092,kernel_size=4,stride=1,padding=0, bias=False),
            nn.BatchNorm2d(4092),
            nn.ReLU(inplace=True),
            # out: 512 x 4 x 4
        )
        self.cnnT2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4092,out_channels=2048,kernel_size=4,stride=2,padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            # out: 256 x 8 x 8
        )
        self.cnnT3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048,out_channels=1024,kernel_size=4,stride=2,padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # out: 128 x 16 x 16
        )
        self.cnnT4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,out_channels=128,kernel_size=4,stride=2,padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # out: 64 x 32 x 32
        )
        self.cnnT5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=3,kernel_size=4,stride=2,padding=1, bias=False),
            nn.Tanh(),
            # out: 3 x 64 x 64
        )
    def forward(self,x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: A 3x64x64 Tensor image as output
        """
        x = self.cnnT1(x)
        x = self.cnnT2(x)
        x = self.cnnT3(x)
        x = self.cnnT4(x)
        x = self.cnnT5(x)
        return x
