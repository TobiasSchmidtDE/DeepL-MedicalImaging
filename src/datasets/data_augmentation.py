import torchvision.transforms as tfs
import cv2
import numpy as np


# more augmentations to explore:
# torchvision.transforms.CenterCrop
# torchvision.transforms.ColorJitter
# torchvision.transforms.RandomCrop
# torchvision.transforms.RandomPerspective
# torchvision.transforms.RandomResizedCrop
# torchvision.transforms.RandomErasing

def augment_image_affine(image, shift=True, rotate=True, zoom=True):
    """
      Random affine transformations from https://github.com/jfhealthcare/Chexpert

      Parameters:
        image (Image)
    """
    img_aug = tfs.Compose([
        tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fillcolor=128)
    ])
    image = img_aug(image)

    return image


def augment_image_eqhist(image):
    """
     Equalize Hist transformations from https://github.com/jfhealthcare/Chexpert

      Parameters:
        image (Image)
    """
    # TODO: fix this function
    raise NotImplementedError("No functional yet.")

    image = np.float32(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def augment_color(image):
    img_aug = tfs.Compose([
        tfs.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    ])

    image = img_aug(image)

    return image


def augment_image(image, augmentation="affine"):
    if augmentation.lower() == "affine":
        return augment_image_affine(image)
    elif augmentation.lower() == "eqhist":
        return augment_image_eqhist(image)
    elif augmentation.lower() == "color":
        return augment_color(image)
    else:
        raise Exception(
            'Unknown augmentation type : {}'.format(augmentation))
