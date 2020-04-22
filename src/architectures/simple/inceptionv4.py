#!/usr/bin/env python
# coding: utf-8

import os
import keras
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


DATASET_FOLDER = 'data/dev_dataset/'

data_train = pd.read_csv(os.path.join(DATASET_FOLDER + 'train.csv'))
data_val = pd.read_csv(os.path.join(DATASET_FOLDER + 'valid.csv'))

# preprocess
data_train = data_train.fillna(0)
data_val = data_val.fillna(0)

# drop lateral images
data_train = data_train[~data_train['Frontal/Lateral'].str.contains("Lateral")]
data_val = data_val[~data_val['Frontal/Lateral'].str.contains("Lateral")]

# drop unrelevant columns
data_train = data_train.drop(
    ["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)
data_val = data_val.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)

# deal with uncertanty (-1) values
data_train = data_train.replace(-1, 1)
data_val = data_val.replace(-1, 1)


train_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255.)

target_size = (224, 224)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=data_train, directory=DATASET_FOLDER, x_col='Path',
    y_col=list(data_train.columns[2:16]),
    class_mode='other', target_size=target_size, batch_size=32
)
valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=data_val, directory=DATASET_FOLDER, x_col='Path',
    y_col=list(data_val.columns[2:16]),
    class_mode='other', target_size=target_size,
    batch_size=32
)


base_model = InceptionV3(include_top=False, weights='imagenet')

# add global pooling and dense output layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction_layer = Dense(14, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=prediction_layer)


# freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False


# compile model
adam = keras.optimizers.Adam()
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


# fit model
num_epochs = 3
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

result = model.fit_generator(generator=train_generator,
                             steps_per_epoch=STEP_SIZE_TRAIN,            
                             validation_data=valid_generator,
                             validation_steps=STEP_SIZE_VALID,
                             epochs=num_epochs)

# save the model
model.save('models/inception/inception-v3.h5')
