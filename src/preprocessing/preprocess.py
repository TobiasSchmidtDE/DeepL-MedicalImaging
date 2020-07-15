import os 
import tensorflow as tf
from pathlib import Path
import pandas as pd
from PIL import Image

# Run this before loading other dependencies, otherwise they might occupy memory on gpu 0 by default and it will stay that way

# Specify which GPU(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

from src.architectures.benchmarks.benchmark_definitions import CHEXPERT_COLUMNS
from src.datasets.generator import ImageDataGenerator


# set directory to save preprocessed files
CHEXPERT_PREPROCESSED_DATASET_DIRECTORY = Path(
    './data/chexpert/preprocessed-256-crop/')

labels = pd.read_csv(
    Path(os.environ.get("CHEXPERT_DATASET_DIRECTORY")) / "train.csv")
generator = ImageDataGenerator(dataset=labels,
                               dataset_folder=Path(os.environ.get(
                                   "CHEXPERT_DATASET_DIRECTORY")),
                               label_columns=CHEXPERT_COLUMNS,
                               path_column="Path",
                               dim=(256, 256),
                               crop=True,
                               batch_size=1)

# iterate through generator and save preprocessed images
for i in range(len(generator)):
    if i % 1000 == 0:
        print("Preprocessing {i} of {l}".format(i=i, l=len(generator)))
    image = generator.__getitem__(i)[0][0]
    path = labels.iloc[i]['Path']

    folderpath = '/'.join(path.split('/')[:-1])
    folderpath = CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / folderpath
    folderpath.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image).save(CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / path)

print("Preprocessing done. Copying labels file")
# copy labels file to new folder
labels.to_csv(CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / 'train.csv')

print("All done, preprocessing completed!")
