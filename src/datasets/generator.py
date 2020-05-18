import pandas as pd
import numpy as np
import keras
import cv2
from skimage.transform import resize
from src.datasets.u_encoding import uencode


class ImageDataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras image classifier
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, dataset, dataset_folder, label_columns, path_column="Path",
                 # TODO: Add support for non-image features (continous and categorical)
                 # conti_feature_columns=[], cat_feature_columns=[],
                 shuffle=True, drop_last=False, batch_size=64, dim=(256, 256), n_channels=3,
                 nan_replacement=0, unc_value=-1, u_enc='uzeroes'):
        """
        Returns a data generator for an image classifier model, that provides batch wise access
        to the data.

        Parameters:
            dataset (pd.dataframe): A pandas daframe with labels and features (paths to images)
                                    for each data sample.
                                    Must contain a column with the paths to the images (relative
                                    to the dataset_folder).
                                    Must contain all columns defined as labels (in label_columns)

            dataset_folder (string|Path): path to the root of the dataset folder
            label_columns (list): names of columns of pathologies we want to use as labels
                        Valid values in label columns are:
                                0 for mentioned as confidently not present,
                                1 for mentioned as confidently present,
                                nan not mentioned,
                                'unc_value' for mentioned as uncertainly present.

            path_column (str): name of the column that contains the relative path from
                            the dataset_folder root directory to each image. (default "Path")

            shuffle (bool): whether to shuffle the data between batches (default True)
            drop_last (bool): wheter to drop the last incomplete batch,
                              if the dataset size is not divisible by the batch size.
                              If False and the size of dataset is not divisible by the batch
                              size, then the last batch will be smaller. (default False)

            batch_size (int): batch size (default 64)
            dim (int): dimension that all images will be resized to (default 256x256)

            n_channels (int): number of channels the image will be converted to (default 3)
                            Note: Every image on disk (source) is expected to be a gray scale image.
                            This parameter only controls the number of channels of the image
                            produced by the generator.

            unc_value (int/str): Value used to indicate uncertainty of pathologies (default -1)
            nan_replacement (int): Value that nan values are replaced with (default 0)
                                Must be a valid value for label columns.
            u_enc (string): style of encoding for uncertainty (default)
                            (values: uzeros, uones, umulticlass)

        Returns:
            generator (DataGenerator): generator with the given specifications
            """

        # check dataset has path column
        if not path_column in dataset.columns:
            raise ValueError(
                path_column + ' is not a column of the dataset dataframe. Columns are: '
                + str(dataset.columns))

        # check that column for paths only contains strings
        if any([(type(path) != str) for path in dataset[path_column]]):
            raise ValueError(
                "Paths to images must be given as string. Dataframe contains non-string values in '"
                + str(path_column) + "' column")

        # check at least one label column is given:
        if len(label_columns) < 1 or label_columns is None:
            raise ValueError(
                "'label_columns' is empty or None, at least one label column must be provided")

        # check all labels are present in dataset dataframe
        for label_column in label_columns:
            if not label_column in dataset.columns:
                raise ValueError(
                    label_column + ' is not a column of the dataset dataframe. Columns are: '
                    + str(dataset.columns))

        # check that all label columns contain valid values {0,1, nan, unc_value}
        valid_values = [0, 1, unc_value]
        for label_column in label_columns:
            if not all(dataset[label_column].isin(valid_values) or dataset[label_column].isna()):
                raise ValueError(
                    label_column + ' contains values which are not valid or NaN. Valid values are '
                    + str(valid_values))

        self.dataset = dataset
        self.dataset_folder = dataset_folder
        self.label_columns = label_columns
        self.path_column = path_column
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.nan_replacement = nan_replacement
        self.unc_value = unc_value
        self.u_enc = u_enc
        self.drop_last = drop_last

        self.on_epoch_end()

    def get_new_index(self):
        """
            Returns a list of ids for the data generation. 
        """
        if self.shuffle:
            return np.random.permutation(len(self.dataset))
        else:
            return range(len(self.dataset))

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        num_batches = len(self.dataset) / self.batch_size
        if self.drop_last:
            return int(np.floor(num_batches))
        else:
            return int(np.ceil(num_batches))

    def data_generation(self, sample_ids):
        """
        Loads one batch of data.

        Parameters:
            sample_ids (integer list): the ids of the samples that should be loaded as part of the batch

        Returns:
            Tupel with list of images and list of labels
        """

        img_paths = self.dataset.iloc[sample_ids][self.path_column].to_numpy()
        img_paths = self.dataset_folder + img_paths

        images = [self.load_image(img_path) for img_path in img_paths]
        labels = self.dataset.iloc[sample_ids][self.label_columns].to_numpy()

        # replace nan values
        labels[np.isnan(labels)] = self.nan_replacement

        # enforce uncertainty encoding strategy
        labels

        return images, labels

    def load_image(self, path):
        """
        Paramter:
            path: the path to the image. Either absolut or relative to repositories root directory.

        Returns a numpy array of the gray scale image

        """
        img = resize(image=cv2.imread(path, cv2.IMREAD_GRAYSCALE),
                     output_shape=self.dim, order=1)

        return np.array(img) if self.n_channels == 1 else np.stack((img,) * self.n_channels, axis=2)

    def __getitem__(self, batch_index):
        """
        Loads specific batch of data

        Paramters:
            batch_index (int): the id of the batch that should be loaded

        Returns:
            Tupel with list of images and list of labels of length batch_size

        """

        start_index = batch_index * self.batch_size
        end_index = (batch_index+1) * self.batch_size

        if ((self.drop_last and end_index > len(self.dataset)) or (start_index > len(self.dataset))):
            raise ValueError("Index out of range! Number of batches exceeded. Only {max_batches} batches available, not {num_batches}.".format(
                max_batches=len(self), num_batches=batch_index))

        return self.__data_generation(self.index[start_index:end_index])

    def _iter_(self):
        """
        An iterable function that samples batches from the dataset.
        """

        for i in range(0, len(self)):
            yield self.__data_generation(self[i])

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """

        self.index = self.get_new_index()
