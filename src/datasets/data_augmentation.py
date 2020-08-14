import torchvision.transforms as tfs


def augment_image(image, shift=True, rotate=True, zoom=True):
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
