import os
import keras
import pandas as pd
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from src.datasets.generator import create_generator

# data loading
DATASET_FOLDER = 'C:/Users/johan/Git/idp-radio-1/data/dev_dataset/'
data = pd.read_csv(os.path.join(DATASET_FOLDER + 'train.csv'), index_col=[0])
data = data[~data['Frontal/Lateral'].str.contains("Lateral")]
data_train, data_val = train_test_split(data, test_size=0.2)

# create generators
columns = ['Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema']
train_datagen = create_generator(data_train, columns=columns, dataset_folder=DATASET_FOLDER)
valid_datagen = create_generator(data_val, columns=columns, dataset_folder=DATASET_FOLDER)

# create model
base_model = DenseNet121(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction_layer = Dense(4, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=prediction_layer)

# compile model
adam = keras.optimizers.Adam()
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# fit model
num_epochs = 3
STEP_SIZE_TRAIN = len(data_train) // train_datagen.batch_size
STEP_SIZE_VALID = len(data_val) // valid_datagen.batch_size
result = model.fit_generator(generator=train_datagen,
                             steps_per_epoch=STEP_SIZE_TRAIN,
                             validation_data=valid_datagen,
                             validation_steps=STEP_SIZE_VALID,
                             epochs=num_epochs)