from skimage.transform import resize
import cv2
from deprecated import deprecated


class Scaler:
    def __init__(self, img, outsize):
        self.img = img
        self.outsize = (outsize, outsize)

    def resize(self, interpolation):
        """
        resize input image with skikit image functionality
        parameters:
        interpolation -- type of interpolation used for resizing
        """
        if interpolation == 'NEAREST':
            out = resize(image=self.img, output_shape=self.outsize, order=0)
        elif interpolation == 'LINEAR':
            out = resize(image=self.img, output_shape=self.outsize, order=1)
        elif interpolation == 'QUADRATIC':
            out = resize(image=self.img, output_shape=self.outsize, order=2)
        elif interpolation == 'CUBIC':
            out = resize(image=self.img, output_shape=self.outsize, order=3)
        else:
            raise NotImplementedError()
        return out

    @deprecated(version='1.0.0', reason="Not as efficient in np.array handling, use scikit image implementation instead")
    def cv2_interpolate(self, interpolation, path):
        img = cv2.imread(path)
        if interpolation == 'NEAREST':
            out = cv2.resize(img, self.outsize, interpolation=cv2.INTER_NEAREST)
        elif interpolation == 'LINEAR':
            out = cv2.resize(img, self.outsize, interpolation=cv2.INTER_LINEAR)
        elif interpolation == 'CUBIC':
            out = cv2.resize(img, self.outsize, interpolation=cv2.INTER_CUBIC)
        else:
            raise NotImplementedError()
        return out
