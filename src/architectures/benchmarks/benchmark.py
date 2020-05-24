import datetime
from pathlib import Path
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
from sklearn.metrics import classification_report
from src.datasets.generator import ImageDataGenerator
from src.utils.save_model import save_model, model_set
from src.preprocessing.split.train_test_split import train_test_split


class Experiment:
    def __init__(self, benchmark, model, model_name, model_version='1'):
        """
        TODO Documentation
        """

        self.benchmark = benchmark
        self.model_name = model_name
        self.model_version = model_version
        self.model_filename = None
        self.model_id = None

        self.model_description = ("Trained {model_name} architecture using predefined benchmark."
                                  ).format(model_name=self.model_name) + benchmark.summary_str()

        self.model = model
        self.model.compile(optimizer=self.benchmark.optimizer,
                           loss=self.benchmark.loss,
                           metrics=self.benchmark.metrics)

        self.train_result = None
        self.evaluation_result = None

    def run(self):
        self.train()
        self.evaluate()
        self.save()

        return self.train_result, self.evaluation_result, self.model_id

    def train(self):
        """ executes training on model """
        traingen = self.benchmark.traingen
        valgen = self.benchmark.valgen

        traingen.reset()
        valgen.reset()

        STEP_SIZE_TRAIN = len(traingen) // traingen.batch_size
        STEP_SIZE_VALID = len(valgen) // valgen.batch_size

        model_dir = self.benchmark.models_dir / self.model_name
        log_dir = model_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=str(log_dir))

        self.train_result = self.model.fit_generator(generator=traingen,
                                                     steps_per_epoch=STEP_SIZE_TRAIN,
                                                     validation_data=valgen,
                                                     validation_steps=STEP_SIZE_VALID,
                                                     epochs=self.benchmark.epochs,
                                                     callbacks=[tensorboard_callback])
        return self.train_result

    def evaluate(self):
        """ evaluates model on test data """
        testgen = self.benchmark.testgen
        testgen.reset()

        STEP_SIZE_TEST = len(testgen) // testgen.batch_size

        predictions = self.model.predict_generator(
            testgen, steps=STEP_SIZE_TEST, verbose=1)

        predictions_bool = (predictions >= 0.5)

        y_pred = np.array(predictions_bool, dtype=int)

        groundtruth_label = testgen.get_encoded_labels()

        report = classification_report(
            groundtruth_label, y_pred, target_names=list(self.benchmark.label_columns))

        score, acc = self.model.evaluate_generator(
            testgen, steps=STEP_SIZE_TEST, verbose=1)

        self.evaluation_result = {
            "report": report,
            "score": score,
            "acc": acc,
            "predictions": predictions,
        }

        return self.evaluation_result

    def save(self):
        """ saves trained model """

        self.model_filename = self.model_name + "_" + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"

        self.model_id = save_model(self.model,
                                   self.train_result.history,
                                   self.model_name,
                                   self.model_filename,
                                   self.model_description,
                                   version=self.model_version)
        if self.evaluation_result != None:
            model_set(self.model_id, 'test',
                      (self.evaluation_result["score"], self.evaluation_result["acc"]))
            model_set(self.model_id, 'classification_report',
                      self.evaluation_result["report"])

        return self.model_id


class Benchmark:
    """
       Wrapper class to standardize experiment execution
    """

    def __init__(self, dataset_folder, label_columns, epochs, models_dir=Path("models/"),
                 optimizer=Adam(), loss='binary_crossentropy', metrics=None,
                 train_labels="train.csv", test_labels=None, split_test_size=0.2,
                 split_valid_size=0.2, split_group='patient_id', split_seed=None, dataset_name=None,
                 shuffle=True, drop_last=False, batch_size=64, dim=(256, 256), n_channels=3,
                 nan_replacement=0, unc_value=-1, u_enc='uzeroes', path_column="Path"):
        """
        TODO: adjust documention

        Parameters:
            dataset_folder (Path): path to the dataset
            columns (list): list of pathologies to be predicted
            epochs (int): numer of epochs for training
            optimizer (keras.optimizer): optimizer used in training
            loss (keras.loss): loss used in training
            metrics (list): metrics to be watched during training
        """

        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.dataset_name = dataset_name
        self.metrics = metrics
        self.dataset_folder = dataset_folder
        self.models_dir = models_dir
        self.label_columns = label_columns
        self.path_column = path_column
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.nan_replacement = nan_replacement
        self.unc_value = unc_value
        self.u_enc = u_enc
        self.drop_last = drop_last

        if self.dataset_name is None:
            self.dataset_name = dataset_folder.parent.name + "_" + dataset_folder.name

        if self.metrics is None:
            self.metrics = ['accuracy']

        if test_labels is None:
            # read all labels from one file and split into train/test/valid
            all_labels = pd.read_csv(self.dataset_folder / train_labels)
            train_labels, test_labels = train_test_split(
                all_labels, test_size=split_test_size, group=split_group, seed=split_seed)
            train_labels, validation_labels = train_test_split(
                train_labels, test_size=split_valid_size, group=split_group, seed=split_seed)
        else:
            # read train/validation labels from one file and test from another
            train_valid_labels = pd.read_csv(
                self.dataset_folder / train_labels)
            test_labels = pd.read_csv(self.dataset_folder / test_labels)
            train_labels, validation_labels = train_test_split(
                train_valid_labels, test_size=split_valid_size, group=split_group, seed=split_seed)

        self.traingen = ImageDataGenerator(dataset=train_labels,
                                           dataset_folder=self.dataset_folder,
                                           label_columns=self.label_columns,
                                           path_column=self.path_column,
                                           shuffle=self.shuffle,
                                           drop_last=self.drop_last,
                                           batch_size=self.batch_size,
                                           n_channels=self.n_channels,
                                           nan_replacement=self.nan_replacement,
                                           unc_value=self.unc_value,
                                           u_enc=self.u_enc)

        self.valgen = ImageDataGenerator(dataset=validation_labels,
                                         dataset_folder=self.dataset_folder,
                                         label_columns=self.label_columns,
                                         path_column=self.path_column,
                                         shuffle=self.shuffle,
                                         drop_last=self.drop_last,
                                         batch_size=self.batch_size,
                                         n_channels=self.n_channels,
                                         nan_replacement=self.nan_replacement,
                                         unc_value=self.unc_value,
                                         u_enc=self.u_enc)

        self.testgen = ImageDataGenerator(dataset=test_labels,
                                          dataset_folder=self.dataset_folder,
                                          label_columns=self.label_columns,
                                          path_column=self.path_column,
                                          shuffle=self.shuffle,
                                          drop_last=self.drop_last,
                                          batch_size=self.batch_size,
                                          n_channels=self.n_channels,
                                          nan_replacement=self.nan_replacement,
                                          unc_value=self.unc_value,
                                          u_enc=self.u_enc)

    def as_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "dataset_folder": self.dataset_folder,
            "models_dir": self.models_dir,
            "epochs": self.epochs,
            "optimizer": self.optimizer.__class__.__name__,
            "loss": self.loss,
            "metrics": self.metrics,
            "label_columns": self.label_columns,
            "path_column": self.path_column,
            "shuffle": self.shuffle,
            "batch_size": self.batch_size,
            "dim": self.dim,
            "n_channels": self.n_channels,
            "nan_replacement": self.nan_replacement,
            "unc_value": self.unc_value,
            "u_enc": self.u_enc,
            "drop_last": self.drop_last,
            "train_num_samples": len(self.traingen.index),
            "valid_num_samples": len(self.valgen.index),
            "test_num_samples": len(self.testgen.index),
        }

    def __str__(self):
        return str(self.as_dict())

    def summary_str(self):
        bench_dict = self.as_dict()
        return ("The benchmark was initialized for the {dataset_name} dataset "
                "with batch size of {batch_size}, shuffel set to {shuffle} "
                "and images rescaled to dimension {dim}.\n"
                "The training was done for {epochs} epochs using the {optimizer} optimizer "
                "and {loss} loss.\nA total of {label_count} labels/pathologies were included "
                "in the training and encoded using the '{u_enc}' method.\n"
                "The traing set included {train_num_samples} "
                "number of sample, the validation set {valid_num_samples}, "
                "and the test set {test_num_samples}. "
                ).format(dataset_name=bench_dict["dataset_name"],
                         batch_size=bench_dict["batch_size"],
                         shuffle=bench_dict["shuffle"],
                         dim=bench_dict["dim"],
                         epochs=bench_dict["epochs"],
                         optimizer=bench_dict["optimizer"],
                         loss=bench_dict["loss"],
                         label_count=len(bench_dict["label_columns"]),
                         u_enc=bench_dict["u_enc"],
                         train_num_samples=bench_dict["train_num_samples"],
                         valid_num_samples=bench_dict["valid_num_samples"],
                         test_num_samples=bench_dict["test_num_samples"],
                         )
