import keras
import cv2
import os
import pandas as pd
import numpy as np
from skimage.transform import resize


def make_dataset(data):
    print('reading dataset')
    x_data, y_data = [], []
    for ind in data.index:
        target = data['Cardiomegaly'][ind]
        if not pd.isna(target):
            img = cv2.imread(os.path.join(dataset_folder + data['Path'][ind]), cv2.IMREAD_GRAYSCALE)
            scaled = resize(image=img, output_shape=(224, 224), order=1)
            x_data.append(np.stack((scaled, scaled, scaled), axis=2))
            if target == -1:
                y_data.append(np.uint8(2))
            else:
                y_data.append(np.uint8(target))
    return np.array(x_data), np.array(y_data)


dataset_folder = "../../../data/dataset/"
chexpert_folder = dataset_folder + "CheXpert-v1.0-small/"

model = keras.applications.densenet.DenseNet121(include_top=True, weights=None, pooling=None, classes=3)

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

data_train = pd.read_csv(os.path.join(dataset_folder + 'train.csv'))
data_train = data_train[:80000]
x_train, y_train = make_dataset(data_train)
data_valid = pd.read_csv(os.path.join(dataset_folder + 'valid.csv'))
x_valid, y_valid = make_dataset(data_valid)

print('# Fit model on training data')
history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=10,
                    validation_data=(x_valid, y_valid))

print('\nhistory dict:', history.history)
model.save('densenet.h5')

# predictions = model.predict(x_test[:10])
# print(predictions)
# print(y_test[:10])
