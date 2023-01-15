from PIL import ImageEnhance
import torch
import torchvision.transforms as T
import random


class AddGaussianNoise(object):  # this is dangerous do not use!!!
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return T.functional.rotate(x, angle)


class EnhanceBrightness(object):
    """This is custom transform class
        It creates bright circle in the center of image
        when initializing you can set
        brightness - float value determine minimal brightness would be product image, should be in <0, 1) <- dimmer <1 , max_bright) <- lighter
        probability - float value (0,1> determines the logically of preforming the transformation
        max_bright - float value (not smaller than bright) maximal brightness of the product image
        when you call it for specific picture it performs call method."""

    def __init__(self, bright: float = 2.5, max_bright: float = 3.0, probability: float = 1.0):
        if bright < 0 or max_bright < bright:
            return
        self.max_bright: float = max_bright
        self.bright: float = bright
        self.probability: float = probability

    def __call__(self, img):
        fate = random.random()  # rand number from (0,1>
        fate_bright = random.random() * abs(self.max_bright - self.bright) + self.bright  # rand number from (0,1>
        if fate <= self.probability:
            return ImageEnhance.Brightness(img).enhance(fate_bright)
        else:
            return img  # do nothing