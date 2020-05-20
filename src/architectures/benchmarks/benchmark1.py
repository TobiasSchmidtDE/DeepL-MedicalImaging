import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.resnet_v2 import ResNet152V2
from src.datasets.generator import ImageDataGenerator
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class BenchmarkOne:
    def __init__(self, model_name, dataset_folder, columns, epochs,
                 optimizer=Adam(), loss='binary_crossentropy', metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        self.epochs = epochs

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.dataset_folder = dataset_folder
        self.train_dataset = pd.read_csv(self.dataset_folder / 'train.csv', index_col=[0])
        self.val_dataset = pd.read_csv(self.dataset_folder / 'val.csv', index_col=[0])
        self.test_dataset = pd.read_csv(self.dataset_folder / 'test.csv', index_col=[0])

        self.traingen = ImageDataGenerator(dataset=self.train_dataset, dataset_folder=self.dataset_folder,
                                           label_columns=columns)
        self.valgen = ImageDataGenerator(dataset=self.val_dataset, dataset_folder=self.dataset_folder,
                                         label_columns=columns)
        self.testgen = ImageDataGenerator(dataset=self.test_dataset, dataset_folder=self.dataset_folder,
                                          label_columns=columns)

    def fit_model(self):
        STEP_SIZE_TRAIN = len(self.train_dataset) // self.traingen.batch_size
        STEP_SIZE_VALID = len(self.val_dataset) // self.valgen.batch_size
        self.model.fit_generator(generator=self.traingen,
                                 steps_per_epoch=STEP_SIZE_TRAIN,
                                 validation_data=self.valgen,
                                 validation_steps=STEP_SIZE_VALID,
                                 epochs=self.epochs)

    def eval_model(self):
        print('todo')

    def save_model(self):
        print('todo')
        # dataset = DATASET_FOLDER.parent.name
        # dataset_version = DATASET_FOLDER.name
        # model_name = "Resnet151V2"
        # model_version = "1"
        # model_filename = model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
        # model_description = model_name + " trained on dataset " + dataset + "_" + dataset_version + "."
