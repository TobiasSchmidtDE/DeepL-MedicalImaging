import pandas as pd
import numpy as np


def create_generator(train_path, val_path):
    print('Creating dataset generator')
    train_df = pd.read_csv(train_path, index_col=[0])
    val_df = pd.read_csv(val_path, index_col=[0])
    partition = {'train': list(train_df.index), 'val': list(val_df.index)}
    columns = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    labels = {key: list(train_df[columns].loc[key]) for key in partition['train']}
    labels.update({key: list(val_df[columns].loc[key]) for key in partition['val']})
    print('Finished reading all labels')


create_generator('../../data/dev_dataset/train.csv', '../../data/dev_dataset/valid.csv')
