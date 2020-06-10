from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
from sklearn.metrics import classification_report
from src.datasets.generator import ImageDataGenerator
from src.utils.save_model import save_model, model_set
from src.preprocessing.split.train_test_split import train_test_split
from src.metrics.metrics import SingleClassMetric
from src.metrics.losses import compute_class_weight


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
            self.model_name = self.model.simple_name + \
                datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.model_simple_name = self.model.simple_name

        self.model_description = ("Trained {model_name} architecture using the "
                                  "'{benchmark_name}' benchmark. "
                                  ).format(model_name=self.model_simple_name,
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

        tensorboard_callback = TensorBoard(log_dir=str(log_dir),
                                           update_freq=int(len(traingen)/100),
                                           histogram_freq=1,
                                           write_graph=False,
                                           embeddings_freq=1)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                   min_delta=0,
                                                                   patience=3,
                                                                   verbose=2,
                                                                   mode='auto',
                                                                   restore_best_weights=False)

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
                                                                  cooldown=1,
                                                                  min_lr=0,)

        terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

        self.train_result = self.model.fit(x=traingen,
                                           steps_per_epoch=len(traingen),
                                           validation_data=valgen,
                                           validation_steps=len(valgen),
                                           epochs=self.benchmark.epochs,
                                           class_weight=self.benchmark.class_weights,
                                           callbacks=[tensorboard_callback,
                                                      early_stopping_callback,
                                                      reduce_lr_callback,
                                                      model_checkpoint_callback,
                                                      terminate_on_nan_callback])
        return self.train_result

    def evaluate(self):
        """ evaluates model on test data """
        testgen = self.benchmark.testgen
        testgen.on_epoch_end()

        self.predictions = self.model.predict(
            testgen, steps=len(testgen), verbose=1)

        predictions_bool = (self.predictions >= 0.5)

        y_pred = np.array(predictions_bool, dtype=int)

        groundtruth_label = testgen.get_labels()

        report = classification_report(
            groundtruth_label, y_pred, target_names=list(self.benchmark.label_columns))

        eval_res = self.model.evaluate(
            x=testgen, steps=len(testgen), verbose=1)

        metric_names = self.benchmark.as_dict()["metrics"]
        eval_metrics = dict(
            zip(["loss"] + metric_names, [float(i) for i in eval_res]))

        self.evaluation_result = {
            "report": report,
            "metrics": eval_metrics,
            "predictions": self.predictions,
            "groundtruth_label": groundtruth_label,
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

        return self.model_id


class Benchmark:
    def __init__(self, dataset_folder, label_columns, name, epochs=10, models_dir=Path("models/"),
                 optimizer=Adam(), loss='binary_crossentropy', single_class_metrics=None,
                 metrics=None, train_labels="train.csv", test_labels=None, split_test_size=0.2,
                 split_valid_size=0.2, split_group='patient_id', split_seed=None, dataset_name=None,
                 shuffle=True, drop_last=True, batch_size=64, dim=(256, 256), crop=False,
                 crop_template=None, view_pos_column="Frontal/Lateral", view_pos_frontal="Frontal",
                 view_pos_lateral="Lateral", n_channels=3, nan_replacement=0, unc_value=-1,
                 u_enc='uzeroes', path_column="Path", path_column_prefix="",
                 use_class_weights=False):
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

            shuffle (bool): (default True)
                    See docs of src.datasets.generator.ImageDataGenerator
            drop_last (bool): (default False)
                    See docs of src.datasets.generator.ImageDataGenerator
            batch_size (int): (default 64)
                    See docs of src.datasets.generator.ImageDataGenerator
            dim (int): (default 256x256)
                    See docs of src.datasets.generator.ImageDataGenerator
            crop_dim (int): (default None)
                    See docs of src.datasets.generator.ImageDataGenerator
            crop_template (dict): (default None)
                    See docs of src.datasets.generator.ImageDataGenerator
            n_channels (int): (default 3)
                    See docs of src.datasets.generator.ImageDataGenerator
            unc_value (int/str): (default -1)
                    See docs of src.datasets.generator.ImageDataGenerator
            nan_replacement (int): (default 0)
                    See docs of src.datasets.generator.ImageDataGenerator
            u_enc (string): (default uzeros)
                    See docs of src.datasets.generator.ImageDataGenerator
            path_column (str): (default "Path")
                    See docs of src.datasets.generator.ImageDataGenerator
            path_column_prefix (str): (default "")
                    See docs of src.datasets.generator.ImageDataGenerator
            crop (bool): (default False)
                    See docs of src.datasets.generator.ImageDataGenerator
            crop_tempalte (dict): (default None)
                    See docs of src.datasets.generator.ImageDataGenerator
            view_pos_column (str): (default "Frontal/Lateral")
                    See docs of src.datasets.generator.ImageDataGenerator
            view_pos_frontal (str): (default "Frontal")
                    See docs of src.datasets.generator.ImageDataGenerator
            view_pos_lateral (str): (default "Lateral")
                    See docs of src.datasets.generator.ImageDataGenerator

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
        self.path_column = path_column
        self.path_column_prefix = path_column_prefix
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dim = dim
        self.crop = crop
        self.view_pos_column = view_pos_column
        self.view_pos_frontal = view_pos_frontal
        self.view_pos_lateral = view_pos_lateral
        self.crop_template = crop_template
        self.n_channels = n_channels
        self.nan_replacement = nan_replacement
        self.unc_value = unc_value
        self.u_enc = u_enc
        self.drop_last = drop_last
        self.use_class_weights = use_class_weights
        self.class_weights = None

        # for each metric in single_class instantiate a metric for each individual pathology
        if self.single_class_metrics is not None:
            for base_metric in self.single_class_metrics:
                for class_id in iter(range(len(label_columns))):
                    class_name = label_columns[class_id].lower().replace(
                        " ", "_")
                    self.metrics += [SingleClassMetric(
                        base_metric, class_id, class_name=class_name)]

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
            # read train/validation labels from one file and test from another
            train_valid_labels = pd.read_csv(
                self.dataset_folder / train_labels)
            test_labels = pd.read_csv(self.dataset_folder / test_labels)
            train_labels, validation_labels = train_test_split(
                train_valid_labels, test_size=split_valid_size, group=split_group, seed=split_seed)

        self.traingen = ImageDataGenerator(dataset=train_labels,
                                           dataset_folder=self.dataset_folder,
                                           label_columns=self.label_columns,
                                           path_column_prefix=self.path_column_prefix,
                                           path_column=self.path_column,
                                           shuffle=self.shuffle,
                                           drop_last=self.drop_last,
                                           batch_size=self.batch_size,
                                           n_channels=self.n_channels,
                                           nan_replacement=self.nan_replacement,
                                           unc_value=self.unc_value,
                                           u_enc=self.u_enc,
                                           dim=self.dim,
                                           crop=self.crop,
                                           view_pos_column=self.view_pos_column,
                                           view_pos_lateral=self.view_pos_lateral,
                                           view_pos_frontal=self.view_pos_frontal,
                                           crop_template=self.crop_template)

        self.valgen = ImageDataGenerator(dataset=validation_labels,
                                         dataset_folder=self.dataset_folder,
                                         label_columns=self.label_columns,
                                         path_column=self.path_column,
                                         path_column_prefix=self.path_column_prefix,
                                         shuffle=self.shuffle,
                                         drop_last=self.drop_last,
                                         batch_size=self.batch_size,
                                         n_channels=self.n_channels,
                                         nan_replacement=self.nan_replacement,
                                         unc_value=self.unc_value,
                                         u_enc=self.u_enc,
                                         dim=self.dim,
                                         crop=self.crop,
                                         view_pos_column=self.view_pos_column,
                                         view_pos_lateral=self.view_pos_lateral,
                                         view_pos_frontal=self.view_pos_frontal,
                                         crop_template=self.crop_template)

        self.testgen = ImageDataGenerator(dataset=test_labels,
                                          dataset_folder=self.dataset_folder,
                                          label_columns=self.label_columns,
                                          path_column=self.path_column,
                                          path_column_prefix=self.path_column_prefix,
                                          shuffle=self.shuffle,
                                          drop_last=self.drop_last,
                                          batch_size=1,
                                          n_channels=self.n_channels,
                                          nan_replacement=self.nan_replacement,
                                          unc_value=self.unc_value,
                                          u_enc=self.u_enc,
                                          dim=self.dim,
                                          crop=self.crop,
                                          view_pos_column=self.view_pos_column,
                                          view_pos_lateral=self.view_pos_lateral,
                                          view_pos_frontal=self.view_pos_frontal,
                                          crop_template=self.crop_template)

        self.positive_weights, self.negative_weights = compute_class_weight(
            self.traingen)
        if self.use_class_weights:
            self.class_weights = {
                i: float(self.positive_weights[i]) for i in range(len(self.positive_weights))}

    def as_dict(self):
        """
        Returns the configuration of this benchmark as a dictionary that is serializable
        """

        metrics = [name for name in self.metrics if isinstance(name, str)]
        metrics += [
            metric.name for metric in self.metrics if not isinstance(metric, str)]
        crop_dims = None
        if self.crop:
            crop_dims = [template["crop_dim"] for t_name,
                         template in self.traingen.template_matcher.templates.items()]
        return {
            "benchmark_name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_folder": str(self.dataset_folder),
            "models_dir": str(self.models_dir),
            "epochs": self.epochs,
            "optimizer": self.optimizer.__class__.__name__,
            "loss": self.loss if isinstance(self.loss, str) else self.loss.name,
            "use_class_weights": self.use_class_weights,
            "positive_weights": [float(i) for i in self.positive_weights.numpy()],
            "negative_weights": [float(i) for i in self.negative_weights.numpy()],
            "metrics": metrics,
            "label_columns": self.label_columns,
            "path_column": self.path_column,
            "path_column_prefix": self.path_column_prefix,
            "shuffle": self.shuffle,
            "batch_size": self.batch_size,
            "dim": self.dim,
            "crop": self.crop,
            "crop_dims": crop_dims,
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

    def summary(self):
        """
        Returns human readable description of the benchmark configuration
        """
        bench_dict = self.as_dict()
        return ("The benchmark was initialized for the {dataset_name} dataset "
                "with batch size of {batch_size}, shuffle set to {shuffle} "
                "and images rescaled to dimension {dim}"
                "and then cropped to {crop_dim}.\n" if self.crop_dim else ".\n"
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
                         crop_dim=bench_dict["crop_dim"],
                         epochs=bench_dict["epochs"],
                         optimizer=bench_dict["optimizer"],
                         loss=bench_dict["loss"],
                         label_count=len(bench_dict["label_columns"]),
                         u_enc=bench_dict["u_enc"],
                         train_num_samples=bench_dict["train_num_samples"],
                         valid_num_samples=bench_dict["valid_num_samples"],
                         test_num_samples=bench_dict["test_num_samples"],
                         )
