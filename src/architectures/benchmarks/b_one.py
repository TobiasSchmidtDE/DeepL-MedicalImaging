import datetime
import pandas as pd
import numpy as np
from src.datasets.generator import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from utils.save_model import save_model, model_set


class BenchmarkOne:
    def __init__(self, model, model_name, dataset_folder, columns, epochs,
                 optimizer=Adam(), loss='binary_crossentropy', metrics=None):
        self.result = None
        if metrics is None:
            metrics = ['accuracy']
        self.epochs = epochs
        self.model = model
        self.model_id = None
        self.model_name = model_name
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
        STEP_SIZE_TRAIN = self.traingen.n // self.traingen.batch_size
        STEP_SIZE_VALID = self.valgen.n // self.valgen.batch_size
        self.result = self.model.fit_generator(generator=self.traingen,
                                 steps_per_epoch=STEP_SIZE_TRAIN,
                                 validation_data=self.valgen,
                                 validation_steps=STEP_SIZE_VALID,
                                 epochs=self.epochs)
        return self.result

    def eval_model(self):
        if self.model_id is None:
            raise LookupError('call save_model before the evaluation')
        STEP_SIZE_TEST = self.testgen.n // self.testgen.batch_size
        self.testgen.reset()
        pred = self.model.predict_generator(self.testgen, steps=STEP_SIZE_TEST, verbose=1)
        pred_bool = (pred >= 0.5)
        y_pred = np.array(pred_bool, dtype=int)

        dtest = self.test_dataset.to_numpy()
        y_true = np.array(dtest[:, slice(1, 15)], dtype=int)
        report = classification_report(
            y_true, y_pred, target_names=list(self.test_dataset.columns[1:15]))
        model_id = model_set(self.model_id, 'classification_report', report)

        score, acc = self.model.evaluate_generator(
            self.testgen, steps=STEP_SIZE_TEST, verbose=1)
        print('Test score:', score)
        print('Test accuracy:', acc)
        model_id = model_set(model_id, 'test', (score, acc))

    def save_model(self):
        dataset = self.dataset_folder.parent.name
        dataset_version = self.dataset_folder.name
        model_filename = self.model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
        model_description = self.model_name + " trained on dataset " + dataset + "_" + dataset_version + "."
        self.model_id = save_model(self.model, self.result.history, self.model_name, model_filename, model_description)
