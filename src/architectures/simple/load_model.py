# Author: See functions

import json
from pathlib import Path
import os
import tensorflow
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy
from src.architectures.benchmarks.benchmark import Benchmark
from src.architectures.benchmarks.benchmark_definitions import Chexpert_Benchmark, Chestxray14_Benchmark, simple_architecture_experiment, generate_benchmarks, METRICS, SINGLE_CLASS_METRICS, CHEXPERT_COLUMNS, CHESTXRAY14_COLUMNS
from src.metrics.losses import WeightedBinaryCrossentropy, compute_class_weight
from src.metrics.metrics import SingleClassMetric, NaNWrapper, F2Score, FBetaScore
from src.architectures.simple.simple_base import SimpleBaseArchitecture
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


def reevaluate_on_datagen(exp, datagen, new_metrics=False):
    # Author: Tobias
    exp.predictions = exp.model.predict(
        datagen, steps=len(datagen), verbose=1)
    exp.groundtruth_label = datagen.get_labels_nonan()

    metric_results = {}
    if new_metrics:
        metrics = []
        for m in exp.benchmark.metrics:
            if isinstance(m, SingleClassMetric):
                class_name = "_".join(m.name.split("_")[1:])
                metrics.append(SingleClassMetric(
                    m.base_metric, m.class_id, class_name=class_name))
            else:
                extra_args = {}
                if hasattr(m, "multi_label"):
                    extra_args["multi_label"] = m.multi_label

                metrics.append(m.__class__(name=m.name, **extra_args))
    else:
        metrics = exp.benchmark.metrics

    for metric in metrics:
        metric.reset_states()
        metric.update_state(tensorflow.constant(
            exp.groundtruth_label), tensorflow.constant(exp.predictions))
        metric_results[metric.name] = metric.result().numpy()

    return metric_results


def reevaluate(exp, new_metrics=False):
    # Author: Tobias
    testgen = exp.benchmark.testgen
    testgen.on_epoch_end()

    return reevaluate_on_datagen(exp, testgen, new_metrics=new_metrics)


def reevaluate_validation(exp, new_metrics=False):
    # Author: Tobias
    valgen = exp.benchmark.valgen
    valgen.on_epoch_end()

    return reevaluate_on_datagen(exp, valgen, new_metrics=new_metrics)


def get_all_experiment_logs(basepath=Path("/srv/idp-radio-1")):
    # Author: Tobias
    logfile_path = basepath / "logs" / "experiment-log.json"
    unvalid_logfile_path = basepath / "logs" / "unvalidated-experiment-log.json"

    with open(logfile_path, 'r') as f:
        logs_data = json.load(f)

    with open(unvalid_logfile_path, 'r') as f:
        unvalid_logs_data = json.load(f)

    experiments = logs_data['experiments'] + unvalid_logs_data['experiments']
    return experiments


def get_experiments_from_logs(names=None, experiment_ids=None):
    # Author: Tobias
    if names is not None:
        experiments = []
        for name in names:
            experiments.append(get_experiment_from_logs(name=name))
        return experiments

    if experiment_ids is not None:
        experiments = []
        for exp_id in experiment_ids:
            experiments.append(get_experiment_from_logs(experiment_id=exp_id))
        return experiments

    return None


def get_experiment_from_logs(name=None, experiment_id=None, version="1", basepath=Path("/srv/idp-radio-1")):
    # Author: Tobias
    experiments = get_all_experiment_logs(basepath=basepath)

    if name is not None:
        experiments_with_name = [
            exp for exp in experiments if exp["name"] == name and exp["version"] == version]
        if len(experiments_with_name) != 0:
            return experiments_with_name[0]

    if experiment_id is not None:
        experiments_with_id = [exp for exp in experiments if exp["id"]
                               == experiment_id and exp["version"] == version]
        if len(experiments_with_id) != 0:
            return experiments_with_id[0]

    return None


def benchmark_from_logs(experiment_dict):
    # Author: Tobias
    benchmark_dict = experiment_dict["benchmark"]

    lr = benchmark_dict["learning_rate"]

    lr_factor = benchmark_dict["lr_factor"] if "lr_factor" in benchmark_dict.keys(
    ) else 1.0

    upsample_factors = benchmark_dict["upsample_factors"] if "upsample_factors" in benchmark_dict.keys(
    ) else None

    optimizer_dict = {
        "Adam": Adam(learning_rate=lr, clipnorm=1),
        "SGD": SGD(learning_rate=lr, clipnorm=1),
    }
    optim = optimizer_dict[benchmark_dict["optimizer"]]

    pos_weights = tensorflow.constant(benchmark_dict["positive_weights"])
    neg_weights = tensorflow.constant(benchmark_dict["negative_weights"])
    loss_dict = {
        "weighted_binary_crossentropy": WeightedBinaryCrossentropy(pos_weights, neg_weights),
        "binary_crossentropy": tensorflow.keras.losses.BinaryCrossentropy(),
        "custom_binary_crossentropy": BinaryCrossentropy()
    }
    loss = loss_dict[benchmark_dict["loss"]]
    transformations = benchmark_dict["transformations"] if "transformations" in benchmark_dict.keys() else {
    }

    return Benchmark(Path(benchmark_dict["dataset_folder"]),
                     benchmark_dict["label_columns"],
                     benchmark_dict["benchmark_name"],
                     epochs=benchmark_dict["epochs"],
                     models_dir=Path(benchmark_dict["models_dir"]),
                     optimizer=optim,
                     lr_factor=lr_factor,
                     loss=loss,
                     single_class_metrics=SINGLE_CLASS_METRICS,
                     metrics=METRICS,
                     train_labels="train.csv",
                     test_labels="test.csv",
                     split_test_size=0.1,
                     split_valid_size=0.05,
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
                     upsample_factors=upsample_factors,
                     preprocess_input_fn=get_preprocessing_for_architecture(
                         experiment_dict["name"]),
                     transformations=transformations)


def get_preprocessing_for_architecture(experiment_name):
    # Author: Tobias
    mapping = {
        "DenseNet121": tensorflow.keras.applications.densenet.preprocess_input,
        "DenseNet169": tensorflow.keras.applications.densenet.preprocess_input,
        "InceptionV3": tensorflow.keras.applications.inception_v3.preprocess_input,
        "ResNet101V2": tensorflow.keras.applications.resnet_v2.preprocess_input,
        "InceptionResNetV2": tensorflow.keras.applications.inception_resnet_v2.preprocess_input,
        "Xception": tensorflow.keras.applications.xception.preprocess_input
    }
    for name in mapping.keys():
        if name in experiment_name:
            return mapping[name]
    print(f"No model function for experiment '{experiment_name}'")
    return None


def get_model_build_function(experiment_name):
    # Author: Tobias
    mapping = {
        "DenseNet121": DenseNet121,
        "DenseNet169": DenseNet169,
        "InceptionV3": InceptionV3,
        "ResNet101V2": ResNet101V2,
        "InceptionResNetV2": InceptionResNetV2,
        "Xception": Xception
    }
    for name in mapping.keys():
        if name in experiment_name:
            return mapping[name]
    print(f"No preproccessing function for model in experiment '{experiment_name}'")
    return None


def rebuild_experiment(experiment_dict, epoch=None, model_folder=Path("/srv/idp-radio-1/models")):
    # Author: Tobias
    benchmark = benchmark_from_logs(experiment_dict)
    architecture_fn = get_model_build_function(experiment_dict["name"])

    weights_file = experiment_dict["filename"]
    model_path = model_folder / experiment_dict["name"]
    if epoch is not None:
        epoch_weight_files = {x.name: x for x in model_path.glob("weights*")}
        for filename in epoch_weight_files.keys():
            if "weights.{:02d}".format(int(epoch)) in filename:
                weights_file = filename
                break

    print(f"Using weights file {weights_file} to load model...")

    weights_path = model_path / weights_file
    if not weights_path.exists():
        raise Exception(f"Weights file '{weights_path}' does not exist ")

    experiment = simple_architecture_experiment(
        benchmark, architecture_fn, benchmark.label_columns)
    experiment.model.load_weights(weights_path)

    return experiment


def difference_test_results(test1, test2):
    # Author: Tobias
    diff = {}
    for metric in test1.keys():
        if metric in test2.keys():
            diff[metric] = abs(test1[metric] - test2[metric])
    return diff


def load_model(ind, weights, validated=True):
    # Author: Johanna
    """
    Re-instantiates benchmark and reloades/compiles model

    Parameters:
        ind (int):
            index of the model in the json
        weights (string):
            name of the weights file
        validated (bool):
            whether to use the validated-json or the unvalidated-json
    """
    if validated:
        with open("logs/experiment-log.json") as f:
            experiments = json.load(f)
    else:
        with open("logs/unvalidated-experiment-log.json") as f:
            experiments = json.load(f)

    experiment = experiments['experiments'][ind]
    experiment_benchmark = experiments['experiments'][ind]['benchmark']
    experiment_benchmark['name'] = experiment_benchmark['benchmark_name']
    experiment_benchmark['dataset_folder'] = Path(
        experiment_benchmark['dataset_folder'])
    experiment_benchmark.pop('benchmark_name', None)
    experiment_benchmark.pop('num_samples_train', None)
    experiment_benchmark.pop('num_samples_validation', None)
    experiment_benchmark.pop('num_samples_test', None)

    if experiment_benchmark['optimizer'] == 'Adam':
        lr = experiment_benchmark['learning_rate']
        experiment_benchmark['optimizer'] = Adam(learning_rate=lr)
        experiment_benchmark.pop('learning_rate', None)
    elif experiment_benchmark['optimizer'] == 'SGD':
        lr = experiment_benchmark['learning_rate']
        experiment_benchmark['optimizer'] = SGD(learning_rate=lr)
        experiment_benchmark.pop('learning_rate', None)
    else:
        raise NotImplementedError()

    posw = tensorflow.constant(
        experiment_benchmark['positive_weights'], dtype=tensorflow.float32)
    negw = tensorflow.constant(
        experiment_benchmark['negative_weights'], dtype=tensorflow.float32)
    if experiment_benchmark['loss'] == "weighted_binary_crossentropy":
        loss = WeightedBinaryCrossentropy(posw, negw)
    elif experiment_benchmark['loss'] == "binary_crossentropy":
        loss = BinaryCrossentropy()
    else:
        raise NotImplementedError()

    experiment_benchmark.pop('positive_weights', None)
    experiment_benchmark.pop('negative_weights', None)
    print(experiment['name'])
    if "DenseNet121" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.densenet.preprocess_input
        model = SimpleBaseArchitecture(DenseNet121, len(
            experiment_benchmark["label_columns"]))
    elif "DenseNet169" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.densenet.preprocess_input
        model = SimpleBaseArchitecture(DenseNet169, len(
            experiment_benchmark["label_columns"]))
    elif "InceptionV3" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.inception_v3.preprocess_input
        model = SimpleBaseArchitecture(InceptionV3, len(
            experiment_benchmark["label_columns"]))
    elif "ResNet101V2" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.resnet_v2.preprocess_input
        model = SimpleBaseArchitecture(ResNet101V2, len(
            experiment_benchmark["label_columns"]))
    elif "InceptionResNetV2" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.inception_resnet_v2.preprocess_input
        model = SimpleBaseArchitecture(InceptionResNetV2, len(
            experiment_benchmark["label_columns"]))
    elif "Xception" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.xception.preprocess_input
        model = SimpleBaseArchitecture(Xception, len(
            experiment_benchmark["label_columns"]))
    else:
        raise NotImplementedError()

    benchmark = Benchmark(**experiment_benchmark)
    testgen = benchmark.testgen
    traingen = benchmark.traingen
    valgen = benchmark.valgen

    basepath = Path(os.getcwd())
    model_name = experiment['name']
    model_folder_path = basepath / "models" / model_name
    weights_path = model_folder_path / weights

    if not weights_path.exists():
        raise Exception(f"Weights file '{weights_path}' does not exist ")
    reconstructed_model = model.load_weights(weights_path)

    return benchmark, reconstructed_model
