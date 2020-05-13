import os
import pandas as pd
import numpy as np
import keras
import cv2
from src.datasets.u_encoding import uencode, uencode_single
from src.preprocessing.rescaling.scaler import Scaler


def create_generator(train_path, val_path, img_size,
                     batch_size, n_channels, columns, u_enc='uzeroes'):
    print('Creating dataset generator')

    train_df = pd.read_csv(train_path, index_col=[0])
    val_df = pd.read_csv(val_path, index_col=[0])
    partition = {'train': list(train_df.index), 'val': list(val_df.index)}

    if type(columns) is not list:
        raise ValueError('columns has to be a list')
    else:
        labels = {key: list(train_df[columns].loc[key]) for key in partition['train']}
        labels.update({key: list(val_df[columns].loc[key]) for key in partition['val']})
        multiple_labels = True if len(columns) > 1 else False

    if multiple_labels:
        labels, num_classes = uencode(u_enc, labels)
    else:
        labels, num_classes = uencode_single(u_enc, labels)

    dataset_folder = train_path.replace('train.csv', '')

    params = {'dim': (img_size, img_size),
              'batch_size': batch_size,
              'n_classes': num_classes,
              'n_channels': n_channels,
              'shuffle': True,
              'dataset_folder': dataset_folder}
    train_gen = DataGenerator(partition['train'], labels, **params)
    val_gen = DataGenerator(partition['val'], labels, **params)
    return train_gen, val_gen


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_ids, labels, batch_size, dim, n_channels,
                 n_classes, shuffle, dataset_folder):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.indexes = np.arange(len(self.list_IDs))
        self.dataset_folder = dataset_folder

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_ids_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""
        X = np.empty(self.batch_size, *self.dim, self.n_channels)
        y = np.empty(self.batch_size, dtype=int)

        for i, ID in enumerate(list_ids_temp):
            img = cv2.imread(os.path.join(self.dataset_folder + ID['Path'], cv2.IMREAD_GRAYSCALE))
            scaled = Scaler(img, self.dim).resize('LINEAR')
            if self.n_channels == 1:
                X[i] = scaled
            elif self.n_channels == 3:
                X[i] = np.stack((scaled, scaled, scaled), axis=2)
            else:
                raise ValueError('Invalid number of channels.')
            y[i] = self.labels[ID]
        return X, y
