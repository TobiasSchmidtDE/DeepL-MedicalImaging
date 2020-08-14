import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from src.preprocessing.cropping.template_matching import TemplateMatcher


# set directory to save preprocessed files
CHEXPERT_PREPROCESSED_DATASET_DIRECTORY = Path(
    './data/chexpert/preprocessed-320-crop/')

# load labels
CHEXPERT_DIR = Path(os.environ.get("CHEXPERT_DATASET_DIRECTORY"))
labels = pd.read_csv(Path(CHEXPERT_DIR) / "train.csv")

# set params
dim = (320, 320)
template_matcher = TemplateMatcher(size=dim)

m = int(labels.shape[0]/1000)
# iterate through dataset and save preprocessed images
for i, row in labels.iterrows():
    if i % m == 0:
        print("Preprocessing {p}% {i} of {l}".format(
            p=i/labels.shape[0], i=i, l=labels.shape[0]))

    # load image
    path = row['Path']
    img = Image.open(str(CHEXPERT_DIR / path))

    if img.mode != "L":
        print(img.mode)
        img = img.convert(mode="L")
    # resize image to dim + 10%
    size = (int(dim[0] * 1.1), int(dim[1] * 1.1))
    img = img.resize(size)
    img = np.array(img).astype(np.float32)

    # crop image to correct dim
    template_type = row['Frontal/Lateral'].lower()
    img = template_matcher.crop(img, template_type)
    img = Image.fromarray(img)

    folderpath = '/'.join(path.split('/')[:-1])
    folderpath = CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / folderpath
    folderpath.mkdir(parents=True, exist_ok=True)

    if img.mode != "L":
        img = img.convert(mode="L")

    img.save(CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / path)

print("Preprocessing done. Copying labels file")
# copy labels file to new folder
labels.to_csv(CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / 'train.csv')

print("All done, preprocessing completed!")
