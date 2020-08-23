import os

from pathlib import Path
import tensorflow as tf
from dotenv import load_dotenv, find_dotenv

from src.architectures.benchmarks.benchmark import Benchmark, Experiment
from tensorflow.keras.optimizers import Adam, SGD
from src.metrics.metrics import F2Score, NaNWrapper
from src.metrics.losses import WeightedBinaryCrossentropy, BinaryCrossentropy, compute_class_weight
from src.architectures.simple.simple_base import SimpleBaseArchitecture

load_dotenv(find_dotenv())

METRICS = [NaNWrapper(tf.keras.metrics.AUC(multi_label=True, name="auc")),
           NaNWrapper(tf.keras.metrics.Precision(name="precision")),
           NaNWrapper(tf.keras.metrics.Recall(name="recall")),
           NaNWrapper(F2Score(name="f2_score")),
           NaNWrapper(tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"))
           ]
SINGLE_CLASS_METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    F2Score(name="f2_score"),
]

"""
METRICS = [tf.keras.metrics.AUC(multi_label=True, name="auc"),
           tf.keras.metrics.Precision(name="precision"),
           tf.keras.metrics.Recall(name="recall"),
           tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")
           ]
SINGLE_CLASS_METRICS = [
]
"""

CHEXPERT_COLUMNS = ['No Finding',
                    'Enlarged Cardiomediastinum',
                    'Cardiomegaly',
                    'Lung Opacity',
                    'Lung Lesion',
                    'Edema',
                    'Consolidation',
                    'Pneumonia',
                    'Atelectasis',
                    'Pneumothorax',
                    'Pleural Effusion',
                    'Pleural Other',
                    'Fracture',
                    'Support Devices']

CHESTXRAY14_COLUMNS = ['Edema',
                       'Atelectasis',
                       'Pneumonia',
                       'Pleural_Thickening',
                       'Cardiomegaly',
                       'Infiltration',
                       'Consolidation',
                       'Fibrosis',
                       'No Finding',
                       'Effusion',
                       'Nodule',
                       'Mass',
                       'Hernia',
                       'Emphysema',
                       'Pneumothorax']


def simple_architecture_experiment(benchmark, base_model_fn, classes, train_last_layer_only=False):
    model = SimpleBaseArchitecture(base_model_fn, len(
        classes), train_last_layer_only=train_last_layer_only)
    experiment = Experiment(benchmark, model)
    return experiment


class Chexpert_Benchmark(Benchmark):
    def __init__(self, name, path=Path(os.environ.get("CHEXPERT_DATASET_DIRECTORY")), classes=None, train_labels="train.csv",
                 split_group='patient_id', path_column="Path", **kwargs):

        if classes is None:
            classes = CHEXPERT_COLUMNS
        super().__init__(path,
                         classes,
                         name,
                         train_labels=train_labels,
                         split_group=split_group,
                         path_column=path_column,
                         **kwargs)


class Chestxray14_Benchmark(Benchmark):
    def __init__(self, name, path=Path(os.environ.get("CHESTXRAY14_DATASET_DIRECTORY")), classes=None, train_labels="meta/data/labels.csv", split_group='Patient ID',
                 path_column="Image Index", path_column_prefix="images/",
                 view_pos_column="View Position", view_pos_frontal="PA", view_pos_lateral="AP",
                 ** kwargs):
        if classes is None:
            classes = CHESTXRAY14_COLUMNS
        super().__init__(path,
                         classes,
                         name,
                         train_labels=train_labels,
                         split_group=split_group,
                         path_column=path_column,
                         path_column_prefix=path_column_prefix,
                         view_pos_column=view_pos_column,
                         view_pos_lateral=view_pos_lateral,
                         view_pos_frontal=view_pos_frontal,
                         **kwargs)


def generate_benchmarks(classes=None, batch_sizes=None, epoch_sizes=None, crop=None, name_suffix="", **kwargs):

    CHEXPERT_BENCHMARKS = {}
    CHESTXRAY14_BENCHMARKS = {}

    BATCH_SIZES = batch_sizes
    EPOCH_SIZES = epoch_sizes
    CROP = crop

    if BATCH_SIZES is None:
        BATCH_SIZES = {
            "Small": 16,
            "Medium": 32,
            "Large": 64,
        }

    if EPOCH_SIZES is None:
        EPOCH_SIZES = {
            "Quick Dev": 3,
            "Dev": 5,
            "Quick Training": 12,
            "Training": 20,
            "Long Training": 50,
        }

    if CROP is None:
        CROP = {
            "C0": False,
            "C1": True
        }

    for _, batch_size in BATCH_SIZES.items():
        for _, epoch_size in EPOCH_SIZES.items():
            for crop_name, crop_val in CROP.items():
                key_suffix = "_E{epoch_size}_B{batch_size}_{crop_name}".format(
                    epoch_size=epoch_size, batch_size=batch_size, crop_name=crop_name)

                if classes is not None:
                    key_suffix += "_N" + str(len(classes))

                key_suffix += name_suffix

                try:
                    CHEXPERT_BENCHMARKS["BCE" + key_suffix] = \
                        Chexpert_Benchmark("Chexpert_BCE"+key_suffix,
                                           classes=classes,
                                           epochs=epoch_size,
                                           batch_size=batch_size,
                                           loss=BinaryCrossentropy(),
                                           metrics=METRICS,
                                           single_class_metrics=SINGLE_CLASS_METRICS,
                                           crop=crop_val,
                                           **kwargs)
                except Exception as err:
                    print("Chexpert_BCE"+key_suffix + " could not be created")
                    print(err)

                try:
                    CHEXPERT_BENCHMARKS["WBCE" + key_suffix] =  \
                        Chexpert_Benchmark("Chexpert_WBCE"+key_suffix,
                                           classes=classes,
                                           epochs=epoch_size,
                                           batch_size=batch_size,
                                           loss=tf.keras.losses.BinaryCrossentropy(),
                                           metrics=METRICS,
                                           single_class_metrics=SINGLE_CLASS_METRICS,
                                           use_class_weights=True,
                                           crop=crop_val,
                                           **kwargs)
                except Exception as err:
                    print("Chexpert_WBCE"+key_suffix + " could not be created")
                    print(err)

                try:
                    CHEXPERT_BENCHMARKS["CWBCE" + key_suffix] =  \
                        Chexpert_Benchmark("Chexpert_CWBCE"+key_suffix,
                                           classes=classes,
                                           epochs=epoch_size,
                                           batch_size=batch_size,
                                           metrics=METRICS,
                                           single_class_metrics=SINGLE_CLASS_METRICS,
                                           crop=crop_val,
                                           **kwargs)
                    # otherweise we get pylint line-too-long in next assignment
                    positive_weights, negative_weights = \
                        CHEXPERT_BENCHMARKS["CWBCE" + key_suffix].positive_weights, \
                        CHEXPERT_BENCHMARKS["CWBCE" + key_suffix].negative_weights
                    CHEXPERT_BENCHMARKS["CWBCE" + key_suffix].loss =  \
                        WeightedBinaryCrossentropy(positive_weights,
                                                   negative_weights)
                except Exception as err:
                    print("Chexpert_CWBCE"+key_suffix + " could not be created")
                    print(err)

                try:
                    CHESTXRAY14_BENCHMARKS["BCE" + key_suffix] =  \
                        Chestxray14_Benchmark("Chestxray_BCE"+key_suffix,
                                              classes=classes,
                                              epochs=epoch_size,
                                              batch_size=batch_size,
                                              loss=tf.keras.losses.BinaryCrossentropy(),
                                              metrics=METRICS,
                                              single_class_metrics=SINGLE_CLASS_METRICS,
                                              crop=crop_val,
                                              **kwargs)
                except Exception as err:
                    print("Chestxray_BCE"+key_suffix + " could not be created")
                    # print(err)

                try:
                    CHESTXRAY14_BENCHMARKS["WBCE" + key_suffix] =  \
                        Chestxray14_Benchmark("Chestxray_WBCE"+key_suffix,
                                              classes=classes,
                                              epochs=epoch_size,
                                              batch_size=batch_size,
                                              metrics=METRICS,
                                              single_class_metrics=SINGLE_CLASS_METRICS,
                                              use_class_weights=True,
                                              crop=crop_val,
                                              **kwargs)
                except Exception as err:
                    print("Chestxray_WBCE"+key_suffix + " could not be created")
                    # print(err)

                try:
                    CHESTXRAY14_BENCHMARKS["CWBCE" + key_suffix] =  \
                        Chestxray14_Benchmark("Chestxray_CWBCE"+key_suffix,
                                              classes=classes,
                                              epochs=epoch_size,
                                              batch_size=batch_size,
                                              metrics=METRICS,
                                              single_class_metrics=SINGLE_CLASS_METRICS,
                                              crop=crop_val,
                                              **kwargs)

                    # otherweise we get pylint line-too-long in next assignment
                    positive_weights, negative_weights = \
                        CHESTXRAY14_BENCHMARKS["CWBCE" + key_suffix].positive_weights, \
                        CHESTXRAY14_BENCHMARKS["CWBCE" +
                                               key_suffix].negative_weights

                    CHESTXRAY14_BENCHMARKS["CWBCE" + key_suffix].loss =  \
                        WeightedBinaryCrossentropy(positive_weights,
                                                   negative_weights)
                except Exception as err:
                    print("Chestxray_CWBCE"+key_suffix +
                          " could not be created")
                    # print(err)

    return CHEXPERT_BENCHMARKS, CHESTXRAY14_BENCHMARKS


def benchmark_from_logs(benchmark_dict):
    lr = benchmark_dict["learning_rate"]
    lr_factor = benchmark_dict["lr_factor"] if "lr_factor" in benchmark_dict.keys() else 1.0
    upsample_factors = benchmark_dict["upsample_factors"] if "upsample_factors" in benchmark_dict.keys() else None
    optimizer_dict = {
        "Adam": Adam(learning_rate=lr, clipnorm=1),
        "SGD": SGD(learning_rate=lr, clipnorm=1),
    }
    optim = optimizer_dict[benchmark_dict["optimizer"]]
    
    pos_weights = tf.constant(benchmark_dict["positive_weights"])
    neg_weights = tf.constant(benchmark_dict["negative_weights"])
    loss_dict = {
        "weighted_binary_crossentropy": WeightedBinaryCrossentropy(pos_weights,neg_weights) ,
        "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(),
        "custom_binary_crossentropy": BinaryCrossentropy()
    }
    loss = loss_dict[benchmark_dict["loss"]]
    
    return Benchmark(Path(benchmark_dict["dataset_folder"]),
                          benchmark_dict["label_columns"],
                          benchmark_dict["benchmark_name"],
                          epochs=benchmark_dict["epochs"],
                          models_dir= Path(benchmark_dict["models_dir"]),
                          optimizer=optim,
                          lr_factor = lr_factor,
                          loss=loss,
                          single_class_metrics=SINGLE_CLASS_METRICS,
                          metrics=METRICS,
                          train_labels="train.csv",
                          test_labels="test.csv",
                          split_test_size= 0.1,
                          split_valid_size = 0.05,
                          split_seed=benchmark_dict["split_seed"],
                          use_class_weights=benchmark_dict["use_class_weights"],
                          path_column=benchmark_dict["path_column"],
                          path_column_prefix=benchmark_dict["path_column_prefix"],
                          shuffle=benchmark_dict["shuffle"],
                          drop_last=benchmark_dict["drop_last"],
                          batch_size=benchmark_dict["batch_size"],
                          dim=benchmark_dict["dim"],
                          n_channels=benchmark_dict["n_channels"],
                          nan_replacement=benchmark_dict["nan_replacement"],
                          unc_value=benchmark_dict["unc_value"],
                          u_enc=benchmark_dict["u_enc"],
                          crop=benchmark_dict["crop"],
                          augmentation=benchmark_dict["augmentation"],
                          upsample_factors= upsample_factors,
                          transformations = [])
    
#CHEXPERT_BENCHMARKS, CHESTXRAY14_BENCHMARKS = generate_benchmarks ()
