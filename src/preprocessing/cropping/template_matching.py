import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import cv2
from skimage.transform import resize

load_dotenv(find_dotenv())

DATASET_FOLDER = Path(os.environ.get('CHEXPERT_DEV_DATASET_DIRECTORY'))

DEFAULT_TEMPLATE = {
    'path': 'CheXpert-v1.0-small/train/patient00165/study2/view1_frontal.jpg',
    'x': (30, 300),
    'y': (30, 300),
    'dim': (320, 327)
}


class TemplateMatcher():
    """ Crops an image to a given size using a template """

    def __init__(self, template=None, matching_method=cv2.TM_CCORR_NORMED, size=(256, 256)):
        """
         Initializes the template matches with a template, a size and a matching method

         Parameters:
            template (dict): A dict containing the path and crop for the template image
            matching_method: The method that is used to match the template to an image.
                             Valid matching methods are listed here:
                             https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#template-matching-in-opencv
            size (tuple): A tuple containing the size of the image to return
        """

        if not template:
            template = DEFAULT_TEMPLATE

        # load template image in the same way as it is loaded in the generator
        img_path = str(DATASET_FOLDER / template['path'])
        # the image is converted to a float32 dtype since template matching does not work
        # with float64
        template_img = resize(image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE),
                              output_shape=template['dim'], order=1).astype('float32')

        x1, x2 = template['x']
        y1, y2 = template['y']
        template_img = template_img[x1:x2, y1:y2]
        self.template = template_img.copy()

        self.matching_method = matching_method
        self.size = size

    def match(self, img):
        """
         Matches the given image to the template and returns the cropped image

         Parameters:
            img (np.array): A numpy array containing an image
         Returns:
            crop_img (np.array): The cropped image
        """
        w, h = self.size

        res = cv2.matchTemplate(img, self.template, self.matching_method)
        min_max = cv2.minMaxLoc(res)
        top_left = min_max[3]

        crop_img = img[top_left[1]:top_left[1]+w, top_left[0]:top_left[0]+h]

        return crop_img
