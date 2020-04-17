import keras
import cv2
import os
import pandas as pd
import numpy as np
from skimage.transform import resize
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)

dataset_folder = "../../../data/dev_dataset/"
chexpert_folder = dataset_folder + "CheXpert-v1.0-small/"

model = keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights=None, pooling=None, classes=3)

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

x_train, y_train = [], []
data_train = pd.read_csv(os.path.join(dataset_folder + 'train.csv'))
data_train = data_train[1:10000]
for ind in data_train.index:
    target = data_train['Cardiomegaly'][ind]
    if not pd.isna(target):
        img = cv2.imread(os.path.join(dataset_folder + data_train['Path'][ind]), cv2.IMREAD_GRAYSCALE)
        scaled = resize(image=img, output_shape=(224, 224), order=1)
        x_train.append(np.stack((scaled, scaled, scaled), axis=2))
        if target == -1:
            y_train.append(2)
        else:
            y_train.append(target)

x_valid, y_valid = [], []
data_valid = pd.read_csv(os.path.join(dataset_folder + 'valid.csv'))
for ind in data_valid.index:
    target = data_valid['Cardiomegaly'][ind]
    if not pd.isna(target):
        img = cv2.imread(os.path.join(dataset_folder + data_valid['Path'][ind]), cv2.IMREAD_GRAYSCALE)
        scaled = resize(image=img, output_shape=(224, 224), order=1)
        x_valid.append(np.stack((scaled, scaled, scaled), axis=2))
        if target == -1:
            y_valid.append(2)
        else:
            y_valid.append(target)


y_train = np.array(y_train)
x_train = np.array(x_train)
y_valid = np.array(y_valid)
x_valid = np.array(x_valid)
print('# Fit model on training data')
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=3,
                    validation_data=(x_valid, y_valid))

print('\nhistory dict:', history.history)


x_test, y_test = [], []
data_test = pd.read_csv(os.path.join(dataset_folder + 'test.csv'))
for ind in data_test.index:
    target = data_test['Cardiomegaly'][ind]
    if not pd.isna(target):
        img = cv2.imread(os.path.join(dataset_folder + data_test['Path'][ind]), cv2.IMREAD_GRAYSCALE)
        scaled = resize(image=img, output_shape=(224, 224), order=1)
        x_test.append(np.stack((scaled, scaled, scaled), axis=2))
        if target == -1:
            y_test.append(2)
        else:
            y_test.append(target)


y_test = np.array(y_test)
x_test = np.array(x_test)
predictions = model.predict(x_test[:10])
print(predictions)
print(y_test[:10])
