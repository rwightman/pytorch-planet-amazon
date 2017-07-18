""" A few fb.resnet.torch like tranforms
Most taken from https://github.com/pytorch/vision/pull/27
"""
import torch
import random
import cv2
import numpy as np


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))


class NormalizeImg:
    """Normalize each image or patch by its own mean/std
    """

    def __init__(self, std_epsilon=.0001):
        self.std_epsilon = std_epsilon

    def __call__(self, img):
        # This should still be a H x W x C Numpy/OpenCv compat image, not a Torch Tensor
        assert isinstance(img, np.ndarray)
        mean, std = cv2.meanStdDev(img)
        mean, std = mean.astype(np.float32), std.astype(np.float32)
        img = img.astype(np.float32)
        img = (img - np.squeeze(mean)) / (np.squeeze(std) + self.std_epsilon)
        return img


class NormalizeImgIn64:
    """Normalize each image or patch by dataset mean/std with math in float64
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean).astype(np.float64)
        self.std = np.array(std).astype(np.float64)

    def __call__(self, img):
        # This should still be a H x W x C Numpy/OpenCv compat image, not a Torch Tensor
        assert isinstance(img, np.ndarray)
        img = img.astype(np.float64)
        img = (img - self.mean) / self.std
        return img.astype(np.float32)


class ToTensor:
    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # handle numpy array
        if img.dtype == np.uint16:
            img = img.astype(np.int32)
            div = 2**16
        elif img.dtype == np.uint32:
            img = img.astype(np.int32)
            div = 2**32
        elif img.dtype == np.int32:
            div = 2**32
        else:
            div = 1.
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        elif isinstance(img, torch.IntTensor):
            return img.float().div(div)
        else:
            return img
