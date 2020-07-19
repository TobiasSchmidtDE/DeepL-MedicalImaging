import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from src.preprocessing.cropping.template_matching import TemplateMatcher


# set directory to save preprocessed files
CHEXPERT_PREPROCESSED_DATASET_DIRECTORY = Path(
    './data/chexpert/preprocessed-256-crop/')

# load labels
CHEXPERT_DIR = Path(os.environ.get("CHEXPERT_DATASET_DIRECTORY"))
labels = pd.read_csv(Path(CHEXPERT_DIR) / "train.csv")[:10]

# set params
dim = (256, 256)
template_matcher = TemplateMatcher(size=dim)


# iterate through dataset and save preprocessed images
for i, row in labels.iterrows():
    if i % 1000 == 0:
        print("Preprocessing {i} of {l}".format(i=i, l=labels.shape[0]))

    # load image
    path = row['Path']
    img = Image.open(str(CHEXPERT_DIR / path))

    # resize image to dim + 10%
    size = (int(dim[0] * 1.1), int(dim[1] * 1.1))
    img = img.resize(size)
    img = np.array(img).astype(np.float32)

    # crop image to correct dim
    template_type = row['Frontal/Lateral'].lower()
    img = template_matcher.crop(img, template_type)
    img = Image.fromarray(img)

    # convert to rbg in order to save it as JPEG
    img = img.convert(mode="RGB")

    folderpath = '/'.join(path.split('/')[:-1])
    folderpath = CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / folderpath
    folderpath.mkdir(parents=True, exist_ok=True)

    img.save(CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / path)

print("Preprocessing done. Copying labels file")
# copy labels file to new folder
labels.to_csv(CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / 'train.csv')

print("All done, preprocessing completed!")
