from skimage.transform import resize
import cv2
import matplotlib.image as mpimg


class Scaler:
    def __init__(self, path, outsize):
        self.path = path
        self.outsize = (outsize, outsize)

    def skimg_resize(self):
        img = mpimg.imread(self.path)
        return resize(img, self.outsize)

    def bilinear_interpolate(self, interpolation):
        img = cv2.imread(self.path)
        if interpolation == 'NEAREST':
            out = cv2.resize(img, self.outsize, interpolation=cv2.INTER_NEAREST)
        elif interpolation == 'LINEAR':
            out = cv2.resize(img, self.outsize, interpolation=cv2.INTER_LINEAR)
        elif interpolation == 'CUBIC':
            out = cv2.resize(img, self.outsize, interpolation=cv2.INTER_CUBIC)
        else:
            raise NotImplementedError()
        return out
