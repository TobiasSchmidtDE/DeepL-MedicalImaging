from skimage import filters as skifilters
from skimage import exposure as skiexposure
import cv2
import numpy as np


class Normalizer:
    """
    Usage:
    1. intantiate the normalizer with a given image using the constructor
        'norm = Normalizer(img)'
    2. apply as many transformations of the image as you'd like
        (e.g. 'norm = norm.apply_windowing(...).apply_unsharp_mask(...)'')
    3. retrieve the normalized image using
        'norm.get_img()'
    """

    def __init__(self, img):
        self.original_img = np.array(img)
        self.img = np.array(img)

    def apply_windowing(self, window=(40, 255)):
        self.img = skiexposure.rescale_intensity(
            self.img, in_range=window)
        return self

    def apply_gaussian_blur(self, kernal_size = 3, sigma=0):
        self.img = cv2.GaussianBlur(self.img, (kernal_size, kernal_size), sigma)
        
        return self

    def apply_hist_equalization(self):
        self.img = cv2.equalizeHist(self.img)
        return self

    def apply_median_filter(self):
        self.img = skifilters.median(self.img)
        return self

    def apply_unsharp_mask(self, radius=1, amount=1):
        self.img = skifilters.unsharp_mask(
            self.img, radius=radius, amount=amount) * 255
        return self

    def set_img(self, img):
        self.img = img
        return self

    def reset(self):
        self.img = self.original_img
        return self

    def get_img(self):
        return self.img.astype('uint8')
