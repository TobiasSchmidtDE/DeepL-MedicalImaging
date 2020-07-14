import os
from pathlib import Path
import pandas as pd
from PIL import Image
from src.architectures.benchmarks.benchmark_definitions import CHEXPERT_COLUMNS
from src.datasets.generator import ImageDataGenerator


# set directory to save preprocessed files
CHEXPERT_PREPROCESSED_DATASET_DIRECTORY = Path(
    './data/chexpert/preprocessed-256-crop/')

labels = pd.read_csv(
    Path(os.environ.get("CHEXPERT_DATASET_DIRECTORY")) / "train.csv")[:20]
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
    image = generator.__getitem__(i)[0][0]
    path = labels.iloc[i]['Path']

    folderpath = '/'.join(path.split('/')[:-1])
    folderpath = CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / folderpath
    folderpath.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image).save(CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / path)


# copy labels file to new folder
labels.to_csv(CHEXPERT_PREPROCESSED_DATASET_DIRECTORY / 'train.csv')
