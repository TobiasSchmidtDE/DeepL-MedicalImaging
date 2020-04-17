import skimage
import cv2
import numpy as np


class Normalizer:
    def __init__(self, img):
        self.original_img = img
        self.img = img

    def apply_windowing (self, window = (40,255)):
        self.img = skimage.exposure.rescale_intensity(self.img, in_range=window) 
        return self

    def apply_gaussian_filter (self, sigma = 1):
        self.img = skimage.filters.gaussian(self.img, sigma) * 255
        self.img = np.uint8(self.img)
        return self
        
    def apply_hist_equalization (self):
        self.img = cv2.equalizeHist(self.img)
        return self

    def apply_median_filter (self):
        self.img = skimage.filters.median(self.img)
        return self

    def apply_unsharp_mask (self, radius = 1, amount = 1):
        self.img = skimage.filters.unsharp_mask(self.img, radius=radius, amount=amount)* 255
        return self
    
    def set_img(self, img):
        self.img = img
        return self
    
    def reset(self):
        self.img = self.original_img
        return self
    
    def get_img(self):
        return self.img
