import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.resnet_v2 import ResNet152V2
from src.datasets.generator import ImageDataGenerator


class SimpleBase:
    def __init__(self, model_name, dataset_folder, columns):
        if model_name == 'inceptionv3':
            self.model = InceptionV3(include_top=False, weights='imagenet')
        elif model_name == 'densenet121':
            self.model = DenseNet121(include_top=False, weights='imagenet')
        elif model_name == 'resnet152v2':
            self.model = ResNet152V2(include_top=False, weights='imagenet')
        # TODO: implement ResNext, other densenet
        else:
            raise NotImplementedError()

        self.dataset_folder = dataset_folder
        train_dataset = pd.read_csv(self.dataset_folder / 'train.csv', index_col=[0])
        val_dataset = pd.read_csv(self.dataset_folder / 'val.csv', index_col=[0])
        test_dataset = pd.read_csv(self.dataset_folder / 'test.csv', index_col=[0])

        self.traingen = ImageDataGenerator(dataset=train_dataset,dataset_folder=self.dataset_folder, label_columns=columns)
        self.valgen = ImageDataGenerator(dataset=val_dataset,dataset_folder=self.dataset_folder, label_columns=columns)
        self.testgen = ImageDataGenerator(dataset=test_dataset,dataset_folder=self.dataset_folder, label_columns=columns)

    def train_model(self):
        print('todo')

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