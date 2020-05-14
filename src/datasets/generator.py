import os
import pandas as pd
import numpy as np
import keras
import cv2
from skimage.transform import resize
from src.datasets.u_encoding import uencode


def create_generator(data, columns, dataset_folder,
                     batch_size=64, n_channels=3,
                     img_size=256, u_enc='uzeroes', shuffle=True):

    """    Returns a generator with the data

    Parameters:
        data (pd.dataframe): dataset dataframe
        dataset_folder (string): path to dataset
        img_size (int): size the images will be resized to (img_size x img_size)
        batch_size (int): batch size
        n_channels (int): number of channels the image will be converted to
        columns (list): columns/pathologies we want to use for training
        u_enc (string): style of encoding for uncertainty
                        (values: uzeros, uones, umulticlass)
        shuffle (bool): whether to shuffle the data between batches

    Returns:
        generator (DataGenerator): generator with the given specifications
        """

    if not isinstance(data, pd.DataFrame):
        raise ValueError('data has to be a dataframe')
    if not isinstance(columns, list) or len(columns) < 1:
        raise ValueError('columns need to be a non-empty list')
    if not isinstance(columns, list):
        raise ValueError('columns has to be a list')
    labels = {key: list(data[columns].loc[key]) for key in data.index}
    labels, num_classes = uencode(u_enc, labels)

    params = {'dim': (img_size, img_size),
              'batch_size': batch_size,
              'n_classes': len(columns),
              'n_channels': n_channels,
              'shuffle': shuffle,
              'dataset_folder': dataset_folder}

    return DataGenerator(data, labels, **params)


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, data, labels, batch_size, dim, n_channels,
                 n_classes, shuffle, dataset_folder):
        """Initialization"""
        self.data = data
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = self.data.index
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dataset_folder = dataset_folder
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data, , samples that are left due to batch size are discarded"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # this is where its wrong
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        if not set(list_ids_temp).issubset(set(self.list_IDs)):
            raise ValueError
        for i, ID in enumerate(list_ids_temp):
            img = cv2.imread(os.path.join(self.dataset_folder + self.data.loc[ID]['Path']), cv2.IMREAD_GRAYSCALE)
            scaled = resize(image=img, output_shape=self.dim, order=1)
            if self.n_channels == 1:
                X[i] = scaled
            elif self.n_channels == 3:
                X[i] = np.stack((scaled, scaled, scaled), axis=2)
            else:
                raise ValueError('Invalid number of channels.')
            y[i] = self.labels[ID]
        return X, y
