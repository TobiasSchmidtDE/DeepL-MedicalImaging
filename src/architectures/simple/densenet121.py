import os
import keras
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.save_model import save_model, model_set


#DATASET_FOLDER = 'data/dataset/'
DATASET_FOLDER = 'data/dev_dataset/'
SEED = 17

# The dev dataset and full dataset have the features in different columns
# we define this variable here to quickly switch between the two
FEATURES_SLICE = slice(2, 16)  # slice(2, 16)
# also, lets define the model name up here so we don't need to scroll all the way down :P
MODEL_NAME = "densenet_121_dev.h5"


data = pd.read_csv(os.path.join(DATASET_FOLDER + 'train.csv'))

# preprocess
data = data.fillna(0)

# drop lateral images
data = data[~data['Frontal/Lateral'].str.contains("Lateral")]

# drop unrelevant columns
data = data.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)

# deal with uncertanty (-1) values
data = data.replace(-1, 1)

np.random.seed(SEED)
data_train, data_test = train_test_split(data, test_size=0.2)
data_train, data_val = train_test_split(data_train, test_size=0.2)

train_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255.)
test_datagen = ImageDataGenerator(rescale=1. / 255.)

target_size = (224, 224)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=data_train,
    directory=DATASET_FOLDER,
    x_col='Path',
    y_col=list(data_train.columns[FEATURES_SLICE]),
    class_mode='other',
    target_size=target_size,
    batch_size=32
)
valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=data_val,
    directory=DATASET_FOLDER,
    x_col='Path',
    y_col=list(data_val.columns[FEATURES_SLICE]),
    class_mode='other',
    target_size=target_size,
    batch_size=32
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=data_test,
    directory=DATASET_FOLDER,
    x_col="Path",
    y_col=list(data_test.columns[FEATURES_SLICE]),
    class_mode="other",
    target_size=target_size,
    shuffle=False,
    batch_size=1
)

base_model = DenseNet121(include_top=False, weights='imagenet')

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
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size


result = model.fit_generator(generator=train_generator,
                             steps_per_epoch=STEP_SIZE_TRAIN,
                             validation_data=valid_generator,
                             validation_steps=STEP_SIZE_VALID,
                             epochs=num_epochs)


# save the model
model_id = save_model(model, result.history, 'densenet',
                      MODEL_NAME, description='')


print("predicting...")
test_generator.reset()
pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)


pred_bool = (pred >= 0.5)
y_pred = np.array(pred_bool, dtype=int)

dtest = data_test.to_numpy()
y_true = np.array(dtest[:, FEATURES_SLICE], dtype=int)
report = classification_report(
    y_true, y_pred, target_names=list(data_test.columns[1:15]))
model_id = model_set(model_id, 'classification_report', report)


score, acc = model.evaluate_generator(
    test_generator, steps=STEP_SIZE_TEST, verbose=1)
print('Test score:', score)
print('Test accuracy:', acc)
model_id = model_set(model_id, 'test', (score, acc))


print("completed training and evaluation")
