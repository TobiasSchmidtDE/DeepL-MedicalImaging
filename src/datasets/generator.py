import pandas as pd
import numpy as np
import keras


def create_generator(train_path, val_path, dim, batch_size, allcoll=True, col_index=None):
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
    elif allcoll is False and col_index is None:
        raise ValueError('If only a specific column is to be used, the column name has to be specified')
    else:
        labels = {key: train_df[col_index].loc[key] for key in partition['train']}
        labels.update({key: val_df[col_index].loc[key] for key in partition['val']})
    print('Finished reading all labels')
    #TODO: encode labels correctly
    #TODO: speficy nclasses correctly
    #TODO: specify n_channels correctly
    params = {'dim': dim,
              'batch_size': batch_size,
              'n_classes': 2,
              'n_channels': 3,
              'shuffle': True}
    return DataGenerator(partition['train'], labels, **params), DataGenerator(partition['val'], labels, **params)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_ids, labels, batch_size=32, dim=(256, 256), n_channels=1,
                 n_classes=2, shuffle=True):
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


create_generator('../../data/dev_dataset/train.csv', '../../data/dev_dataset/valid.csv')
