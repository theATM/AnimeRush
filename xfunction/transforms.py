from PIL import ImageEnhance
import torch
import torchvision.transforms as T
import random


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