import os
from pathlib import Path
import tensorflow as tf
from dotenv import load_dotenv, find_dotenv

from src.architectures.benchmarks.benchmark import Benchmark
from src.metrics.metrics import F2Score
from src.metrics.losses import WeightedBinaryCrossentropy


load_dotenv(find_dotenv())

METRICS = [tf.keras.metrics.AUC(multi_label=True, name="auc"),
           tf.keras.metrics.Precision(name="precision"),
           tf.keras.metrics.Recall(name="recall"),
           F2Score(name="f2_score"),
           tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")
           ]
SINGLE_CLASS_METRICS = [
    # tf.keras.metrics.BinaryAccuracy(name="accuracy")
    # tf.keras.metrics.AUC(name="auc")
]

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


class Chexpert_Benchmark(Benchmark):
    def __init__(self, name, train_labels="train.csv",
                 split_group='patient_id', path_column="Path", **kwargs):

        super().__init__(Path(os.environ.get("CHEXPERT_DATASET_DIRECTORY")),
                         CHEXPERT_COLUMNS,
                         name,
                         train_labels=train_labels,
                         split_group=split_group,
                         path_column=path_column,
                         **kwargs)


class Chestxray14_Benchmark(Benchmark):
    def __init__(self, name, train_labels="meta/data/labels.csv", split_group='Patient ID',
                 path_column="Image Index", path_column_prefix="images/", **kwargs):
        super().__init__(Path(os.environ.get("CHESTXRAY14_DATASET_DIRECTORY")),
                         CHESTXRAY14_COLUMNS,
                         name,
                         train_labels=train_labels,
                         split_group=split_group,
                         path_column=path_column,
                         path_column_prefix=path_column_prefix,
                         **kwargs)


CHEXPERT_BENCHMARKS = {}
CHESTXRAY14_BENCHMARKS = {}

BATCH_SIZES = {
    "Small": 16,
    "Medium": 32,
    "Large": 64,
}

EPOCH_SIZES = {
    "Quick Dev": 3,
    "Dev": 10,
    "Quick Training": 20,
    "Training": 50,
    "Full Training": 100,
}


for batch_name, batch_size in BATCH_SIZES.items():
    for epoch_name, epoch_size in EPOCH_SIZES.items():
        key_suffix = "_E{epoch_size}_B{batch_size}".format(
            epoch_size=epoch_size, batch_size=batch_size)
        name_suffix = "{epoch_name} Epochs and {batch_name} Batches".format(
            epoch_name=epoch_name, batch_name=batch_name)

        CHEXPERT_BENCHMARKS["BCE" + key_suffix] = \
            Chexpert_Benchmark("Chexpert BCE "+name_suffix,
                               epochs=epoch_size,
                               batch_size=batch_size,
                               loss=tf.keras.losses.BinaryCrossentropy(),
                               metrics=METRICS,
                               single_class_metrics=SINGLE_CLASS_METRICS)

        CHEXPERT_BENCHMARKS["WBCE" + key_suffix] =  \
            Chexpert_Benchmark("Chexpert Weighted BCE "+name_suffix,
                               epochs=epoch_size,
                               batch_size=batch_size,
                               loss=tf.keras.losses.BinaryCrossentropy(),
                               metrics=METRICS,
                               single_class_metrics=SINGLE_CLASS_METRICS,
                               use_class_weights=True)

        CHEXPERT_BENCHMARKS["CWBCE" + key_suffix] =  \
            Chexpert_Benchmark("Chexpert Custom Weighted BCE "+name_suffix,
                               epochs=epoch_size,
                               batch_size=batch_size,
                               metrics=METRICS,
                               single_class_metrics=SINGLE_CLASS_METRICS)
        CHEXPERT_BENCHMARKS["CWBCE" + key_suffix].loss =  \
            WeightedBinaryCrossentropy(CHEXPERT_BENCHMARKS["CWBCE" + key_suffix].positive_weights,
                                       CHEXPERT_BENCHMARKS["CWBCE" + key_suffix].negative_weights)

        CHESTXRAY14_BENCHMARKS["BCE" + key_suffix] =  \
            Chestxray14_Benchmark("Chestxray NIH 14 BCE "+name_suffix,
                                  epochs=epoch_size,
                                  batch_size=batch_size,
                                  loss=tf.keras.losses.BinaryCrossentropy(),
                                  metrics=METRICS,
                                  single_class_metrics=SINGLE_CLASS_METRICS)

        CHESTXRAY14_BENCHMARKS["WBCE" + key_suffix] =  \
            Chestxray14_Benchmark("Chestxray NIH 14 Weighted BCE "+name_suffix,
                                  epochs=epoch_size,
                                  batch_size=batch_size,
                                  metrics=METRICS,
                                  single_class_metrics=SINGLE_CLASS_METRICS,
                                  use_class_weights=True)

        CHESTXRAY14_BENCHMARKS["CWBCE" + key_suffix] =  \
            Chestxray14_Benchmark("Chestxray NIH 14 Custom Weighted BCE "+name_suffix,
                                  epochs=epoch_size,
                                  batch_size=batch_size,
                                  metrics=METRICS,
                                  single_class_metrics=SINGLE_CLASS_METRICS)

        # otherweise we get pylint line-too-long in next assignment
        positive_weights, negative_weights = \
            CHESTXRAY14_BENCHMARKS["CWBCE" + key_suffix].positive_weights, \
            CHESTXRAY14_BENCHMARKS["CWBCE" + key_suffix].negative_weights

        CHESTXRAY14_BENCHMARKS["CWBCE" + key_suffix].loss =  \
            WeightedBinaryCrossentropy(positive_weights,
                                       negative_weights)
