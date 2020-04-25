import pandas as pd
import numpy as np
import keras
from src.datasets.u_encoding import uencode, uencode_single


def create_generator(train_path, val_path, dim, batch_size, allcoll=True, col_index=None, u_enc='uzeroes'):
    print('Creating dataset generator')

    train_df = pd.read_csv(train_path, index_col=[0])
    val_df = pd.read_csv(val_path, index_col=[0])
    partition = {'train': list(train_df.index), 'val': list(val_df.index)}

    if allcoll is True:
        columns = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                   'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        labels = {key: list(train_df[columns].loc[key]) for key in partition['train']}
        labels.update({key: list(val_df[columns].loc[key]) for key in partition['val']})
        multiple_labels = True
    elif allcoll is False and col_index is None:
        raise ValueError('If only a specific column is to be used, the column name has to be specified')
    else:
        labels = {key: train_df[col_index].loc[key] for key in partition['train']}
        labels.update({key: val_df[col_index].loc[key] for key in partition['val']})
        multiple_labels = False

    if multiple_labels:
        labels, num_classes = uencode(u_enc, labels)
    else:
        labels, num_classes = uencode_single(u_enc, labels)

    print(labels)
    params = {'dim': dim,
              'batch_size': batch_size,
              'n_classes': num_classes,
              'n_channels': 3,
              'shuffle': True}

    # TODO: specify n_channels correctly
    return DataGenerator(partition['train'], labels, **params), DataGenerator(partition['val'], labels, **params)


# TODO: reimplement reading of pictures, add scaling and rgb
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_ids, labels, batch_size, dim, n_channels,
                 n_classes, shuffle):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.indexes = np.arange(len(self.list_IDs))

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
            X[i, ] = np.load('data/' + ID + '.npy')
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


create_generator('../../data/dev_dataset/train.csv', '../../data/dev_dataset/valid.csv', dim=(256, 256), batch_size=16)
