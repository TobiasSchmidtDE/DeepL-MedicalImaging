import numpy as np
from PIL import Image
from dotenv import load_dotenv, find_dotenv
import cv2

load_dotenv(find_dotenv())

DEFAULT_TEMPLATE = {
    'frontal':      {
        'path': 'src/preprocessing/cropping/templates/chexpert-frontal.jpg',
        'x': (30, 300),
        'y': (30, 300),
        'dim': (320, 327)
    },
    'lateral': {
        'path': 'src/preprocessing/cropping/templates/chexpert-lateral.jpg',
        'x': (30, 300),
        'y': (20, 290),
        'dim': (349, 320)
    }
}


class TemplateMatcher():
    # Author: Kristian
    """ Crops an image to a given size using a template """

    def __init__(self, template_conf=None, matching_method=cv2.TM_CCORR_NORMED, size=(256, 256)):
        """
         Initializes the template matches with a template, a size and a matching method

         Parameters:
            template_conf (dict): A dict containing the path and crop for the template image
                             path: the path to the template image
                             x (tuple): a tuple containing the start and end of the crop on the
                                        x-axis
                             y (tuple): a tuple containing the start and end of the crop on the
                                        y-axis
                             dim: the original dimension of the image, or the dimension on which
                                  the crop should be applied
            matching_method: The method that is used to match the template to an image.
                             Valid matching methods are listed here:
                             https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#template-matching-in-opencv
        """

        if not template_conf:
            template_conf = DEFAULT_TEMPLATE

        self.templates = {}
        for template_type in template_conf.keys():
            self.templates[template_type] = template_conf[template_type]

            # load template image in the same way as it is loaded in the generator
            img_path = self.templates[template_type]['path']
            # the image is converted to a float32 dtype since template matching does not work
            # with float64
            template_img = np.array(Image.open(img_path)
                                    .resize(self.templates[template_type]['dim'])).astype('float32')

            x1, x2 = self.templates[template_type]['x']
            y1, y2 = self.templates[template_type]['y']
            template_img = template_img[x1:x2, y1:y2]
            self.templates[template_type]["img"] = template_img.copy()

        self.size = size
        self.matching_method = matching_method

    def crop(self, img, template_type='frontal'):
        """
         Matches the given image to the template and returns the cropped image

         Parameters:
            img (np.array): A numpy array containing an image
         Returns:
            crop_img (np.array): The cropped image
        """
        w, h = self.size

        res = cv2.matchTemplate(img,
                                self.templates[template_type]["img"],
                                self.matching_method)
        min_max = cv2.minMaxLoc(res)
        top_left = min_max[3]

        crop_img = img[top_left[1]:top_left[1]+w, top_left[0]:top_left[0]+h]

        return crop_img
