from datetime import datetime
from pathlib import Path
import os
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
from sklearn.metrics import classification_report
from src.datasets.generator import ImageDataGenerator
from src.utils.save_model import save_model, model_set
from src.preprocessing.split.train_test_split import train_test_split
from src.metrics.metrics import SingleClassMetric, NaNWrapper
from src.metrics.losses import WeightedBinaryCrossentropy, BinaryCrossentropy, compute_class_weight
from src.metrics.custom_callbacks import CustomTensorBoard


class Experiment:
    def __init__(self, benchmark, model, model_name=None, model_version='1'):
        """
        Intiantiates an experiment to train and evaluate a given model.

        Parameters:
            benchmark (src.architecures.benchmark.Benchmark):
                    The benchmark configuration to be used providing model
                    and dataset configurations.
            model (tf.keras.Model):
                    An valid keras model instance to be trained.
            model_name (str):
                    Name of the model to be used for documentation/logging.
                    If none the name will be derived from the simple_name attribute
                    of the provided model instance
            model_version (str): (default "1")
                    The version number of the model
        Returns:
            experiment (Experiment):
                    A experiment instance with the given specifications
        """

        self.benchmark = benchmark
        self.model = model
        self.model_name = model_name
        self.model_simple_name = model_name
        self.model_version = model_version
        self.model_filename = None
        self.model_id = None

        if self.model_name is None:
            self.model_name = self.model.simple_name.replace(" ", "_") + "_" +\
                self.benchmark.name.replace(" ", "_")

        self.model_description = ("Trained {model_name} architecture using the "
                                  "'{benchmark_name}' benchmark. "
                                  ).format(model_name=self.model.simple_name,
                                           benchmark_name=benchmark.name)
        self.model_description += benchmark.summary()

        self.model.compile(optimizer=self.benchmark.optimizer,
                           loss=self.benchmark.loss,
                           metrics=self.benchmark.metrics)

        self.train_result = None
        self.evaluation_result = None
        self.predictions = None

    def run(self):
        """
        Trains, evaluates and saves the model
        """

        self.train()
        self.evaluate()
        self.save()

        return self.train_result, self.evaluation_result, self.model_id

    def train(self):
        """ executes training on model """
        traingen = self.benchmark.traingen
        valgen = self.benchmark.valgen

        traingen.on_epoch_end()
        valgen.on_epoch_end()

        model_dir = self.benchmark.models_dir / self.model_name
        checkpoint_filepath = str(
            model_dir / "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

        log_dir = model_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        tensorboard_callback = CustomTensorBoard(log_dir=str(log_dir),
                                           update_freq=int(len(traingen)/200),
                                           histogram_freq=0,
                                           write_graph=False,
                                           profile_batch=0,
                                           embeddings_freq=0)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                   min_delta=0,
                                                                   patience=5,
                                                                   verbose=2,
                                                                   mode='auto',
                                                                   restore_best_weights=True)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                       monitor='val_loss',
                                                                       verbose=2,
                                                                       save_best_only=False,
                                                                       save_weights_only=False,
                                                                       mode='auto',
                                                                       save_freq='epoch')

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                  factor=0.5,
                                                                  patience=2,
                                                                  verbose=0,
                                                                  mode='auto',
                                                                  min_delta=0.0001,
                                                                  cooldown=0,
                                                                  min_lr=0,)
        
        scheduler = lambda epoch, lr: lr if epoch == 0 else lr * self.benchmark.lr_factor 
        lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()
        
        class_weights = None
        if self.benchmark.use_class_weights:
            class_weights = { i: float(self.benchmark.class_weights[i]) for i in range(len(self.benchmark.class_weights))}
            print("Train with class weights: ", class_weights)
        self.train_result = self.model.fit(x=traingen,
                                           steps_per_epoch=len(traingen),
                                           validation_data=valgen,
                                           validation_steps=len(valgen),
                                           epochs=self.benchmark.epochs,
                                           class_weight= class_weights,
                                           callbacks=[tensorboard_callback,
                                                      early_stopping_callback,
                                                      model_checkpoint_callback,
                                                      #terminate_on_nan_callback,
                                                      lr_scheduler_callback,])
        return self.train_result

    def evaluate(self):
        """ evaluates model on test data """
        testgen = self.benchmark.testgen
        testgen.on_epoch_end()

        self.predictions = self.model.predict(
            testgen, steps=len(testgen), verbose=1)

        predictions_bool = (self.predictions >= 0.5)

        self.y_pred = np.array(predictions_bool, dtype=int)

        self.groundtruth_label = testgen.get_labels_nonan()

        self.report = classification_report(
            self.groundtruth_label, self.y_pred, target_names=list(self.benchmark.label_columns))

        self.eval_res = self.model.evaluate(
            x=testgen, steps=len(testgen), verbose=1)

        metric_names = self.benchmark.as_dict()["metrics"]
        eval_metrics = dict(
            zip(["loss"] + metric_names, [float(i) for i in self.eval_res if not math.isnan(float(i))]))

        self.evaluation_result = {
            "report": self.report,
            "metrics": eval_metrics,
            "predictions": self.predictions,
            "groundtruth_label": self.groundtruth_label,
        }

        return self.evaluation_result

    def save(self, upload=True):
        """ saves trained model """

        self.model_filename = self.model_name + "_" + \
            datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"

        self.model_id = save_model(self.model,
                                   self.train_result.history,
                                   self.model_name,
                                   self.model_filename,
                                   self.model_description,
                                   version=self.model_version,
                                   upload=upload)

        model_set(self.model_id, 'benchmark',
                  self.benchmark.as_dict())

        if not self.evaluation_result is None:
            model_set(self.model_id, 'test', self.evaluation_result["metrics"])
            model_set(self.model_id, 'classification_report',
                      self.evaluation_result["report"])
        
        # Save predictions 
        CURRENT_WORKING_DIR = Path(os.getcwd())
        basepath = CURRENT_WORKING_DIR
        # path main directory
        if basepath.name != "idp-radio-1":
            basepath = basepath.parent.parent
        folderpath = basepath / 'models' / self.model_name
        # make sure path exists, ceate one if necessary
        Path(folderpath).mkdir(parents=True, exist_ok=True)
        
        np.savetxt(folderpath / "predictions_probs.csv", self.predictions, delimiter=";")
        np.savetxt(folderpath / "predictions_classes.csv", self.y_pred, delimiter=";")
           
        return self.model_id


class Benchmark:
    def __init__(self, dataset_folder, label_columns, name, epochs=10, models_dir=Path("models/"),
                 optimizer=Adam(), lr_factor = 1.0 , loss=tf.keras.losses.BinaryCrossentropy(), single_class_metrics=[],
                 metrics=None, train_labels="train.csv", test_labels=None, split_test_size=0.2,
                 split_valid_size=0.2, split_group='patient_id', split_seed=None, dataset_name=None,
                 use_class_weights=False, **datagenargs):
        """
        Instaniates a benchmark that can be provided as basis of an
        src.architecures.benchmark.Experiment. Provides these experiments with the same
        training/validation/test data as well as model configurations.

        Parameters:
            dataset_folder (Pathlib Path):
                    See docs of src.datasets.generator.ImageDataGenerator
            label_columns (list):
                    See docs of src.datasets.generator.ImageDataGenerator
            name (str):
                    The human readable name of the benchmark for documentation purposes.
            epochs (int): (default 10)
                    Number of epochs to run the training for
            models_dir (Pathlib Path): (default: Path("models/"))
                    The directory where the models should be saved to

            optimizer (tf.keras.optimizers.Optimizer):  (default: Adam())
                    A valid keras optimizer instance. Learning rate and other parameters
                    can be provided to the Optimizer class when initializing.

            loss (str | tf.keras.losses.Loss):(default: "binary_crossentropy")
                    Either the name or an instance of a valid keras loss.

            metrics (list tf.keras.metrics.Metric): (default [tf.keras.metrics.AUC()])
                    A list of metrics to be evaluated after each epoch. List can contain
                    Either the name or an instance of a valid keras metrics.

            single_class_metrics (list tf.keras.metrics.Metric): (default None)
                    A list of metrics that should be evaluated on each class/pathology individually.

            use_class_weights (bool): (default false)
                    Whether the model trainig (.fit) should be supplied with class_weight factor.
                    Class_weights will be automatically calculated based on the distriubtion of
                    labels to counteract any imbalances in the training data.

            train_labels (str): (default "train.csv")
                    The name of the CSV file containing the labels and features
                    (paths to images) for each data sample to be used for training.
                    These will be split into test, train and validation set
            test_labels (str): (default None)
                    The name of the CSV file for the test data. If given, the samples given
                    in the train_labels file will be used for train and validation only.
                    The samples of the test_labels file will then be used as the test set.
            split_test_size (float): (default 0.2)
                    The relative size of the test set in the train/test split. Only used
                    when no test_labels are provided separately
            split_valid_size (float): (default 0.2)
                    The relative size of the validation set in the train/validation split
                    When no test_labels are provided. The data is first split in train and test.
                    Then the train data is split again in train and validation.
            split_group (bool): (default "patient_id")
                    See docs of src.preprocessing.split.train_test_split.train_test_split
            split_seed (bool): (default None)
                    See docs of src.preprocessing.split.train_test_split.train_test_split

        Returns:
            benchmark (Benchmark):
                    A benchmark given the specifications with data generators already initialized
        """

        self.name = name
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.dataset_name = dataset_name
        # use [:] to make a copy by value instead of reference
        self.single_class_metrics = single_class_metrics[:]
        self.metrics = metrics[:]
        self.dataset_folder = dataset_folder
        self.models_dir = models_dir
        self.label_columns = label_columns
        self.use_class_weights = use_class_weights
        self.class_weights = None
        self.split_seed = split_seed
        self.lr = round(float(self.optimizer.learning_rate), 8)
        self.lr_factor = lr_factor

        # for each metric in single_class instantiate a metric for each individual pathology
        if self.single_class_metrics is not None:
            for base_metric in self.single_class_metrics:
                for class_id in iter(range(len(label_columns))):
                    class_name = label_columns[class_id].lower().replace(
                        " ", "_")
                    self.metrics += [NaNWrapper(SingleClassMetric(
                        base_metric, class_id, class_name=class_name))]

                    
                    
        if self.dataset_name is None:
            self.dataset_name = dataset_folder.parent.name + "_" + dataset_folder.name

        if self.metrics is None:
            self.metrics = [tf.keras.metrics.AUC()]

        if test_labels is None:
            # read all labels from one file and split into train/test/valid
            all_labels = pd.read_csv(self.dataset_folder / train_labels)
            train_labels, test_labels = train_test_split(
                all_labels, test_size=split_test_size, group=split_group, seed=split_seed)
            train_labels, validation_labels = train_test_split(
                train_labels, test_size=split_valid_size, group=split_group, seed=split_seed)
        else:
            # read train and valid labels from one file and test from another.
            train_labels = pd.read_csv(self.dataset_folder / train_labels)
            train_labels, validation_labels = train_test_split(
                train_labels, test_size=split_valid_size, group=split_group, seed=split_seed)
            test_labels = pd.read_csv(self.dataset_folder / test_labels)

        self.traingen = ImageDataGenerator(train_labels,
                                           self.dataset_folder,
                                           self.label_columns,
                                           **datagenargs)

        datagenargs["augmentation"] = None
        datagenargs["upsample_factors"] = None        
        self.valgen = ImageDataGenerator(validation_labels,
                                           self.dataset_folder,
                                           self.label_columns,
                                           **datagenargs)
        
        datagenargs["batch_size"] = 1
        self.testgen = ImageDataGenerator(test_labels,
                                          self.dataset_folder,
                                          self.label_columns,
                                          **datagenargs)

        self.class_weights , self.positive_weights, self.negative_weights = compute_class_weight(
            self.traingen)

    def as_dict(self):
        """
        Returns the configuration of this benchmark as a dictionary that is serializable
        """

        metrics = [name for name in self.metrics if isinstance(name, str)]
        metrics += [
            metric.name for metric in self.metrics if not isinstance(metric, str)]
        return {
            "benchmark_name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_folder": str(self.dataset_folder),
            "models_dir": str(self.models_dir),
            "epochs": self.epochs,
            "optimizer": self.optimizer.__class__.__name__,
            "learning_rate": self.lr,
            "lr_factor": self.lr_factor,
            "loss": self.loss if isinstance(self.loss, str) else self.loss.name,
            "use_class_weights": self.use_class_weights,
            "class_weights": [float(i) for i in self.class_weights.numpy()],
            "positive_weights": [float(i) for i in self.positive_weights.numpy()],
            "negative_weights": [float(i) for i in self.negative_weights.numpy()],
            "metrics": metrics,
            "label_columns": self.label_columns,
            "path_column": self.traingen.path_column,
            "path_column_prefix": self.traingen.path_column_prefix,
            "upsample_factors": self.traingen.upsample_factors,
            "shuffle": self.traingen.shuffle,
            "batch_size": self.traingen.batch_size,
            "dim": self.traingen.dim,
            "crop": self.traingen.crop,
            "transformations": self.traingen.transformations,
            "augmentation": self.traingen.augmentation,
            "n_channels": self.traingen.n_channels,
            "nan_replacement": self.traingen.nan_replacement,
            "unc_value": self.traingen.unc_value,
            "u_enc": self.traingen.u_enc,
            "drop_last": self.traingen.drop_last,
            "num_samples_train": len(self.traingen.index),
            "num_samples_validation": len(self.valgen.index),
            "num_samples_test": len(self.testgen.index),
            "split_seed": self.split_seed
        }

    def __str__(self):
        return str(self.as_dict())

    def summary(self):
        """
        Returns human readable description of the benchmark configuration
        """
        bench_dict = self.as_dict()
        return ("The benchmark was initialized for the {dataset_name} dataset "
                "with batch size of {batch_size}, shuffle set to {shuffle} "
                "and images rescaled " +
                ("and cropped" if self.traingen.crop else "")
                + "to dimension {dim}.\n"
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
                         train_num_samples=bench_dict["num_samples_train"],
                         valid_num_samples=bench_dict["num_samples_validation"],
                         test_num_samples=bench_dict["num_samples_test"],
                         )
