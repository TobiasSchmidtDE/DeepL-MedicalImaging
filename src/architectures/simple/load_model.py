import json
from pathlib import Path

import tensorflow
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy
from src.architectures.benchmarks.benchmark import Benchmark
from src.metrics.losses import WeightedBinaryCrossentropy, compute_class_weight
from src.metrics.metrics import SingleClassMetric, NaNWrapper, F2Score, FBetaScore
from src.architectures.simple.simple_base import SimpleBaseArchitecture
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

def load_model(ind, weights, validated=True):
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
    experiment_benchmark['dataset_folder'] = Path(experiment_benchmark['dataset_folder'])
    experiment_benchmark.pop('benchmark_name', None)
    experiment_benchmark.pop('num_samples_train', None)
    experiment_benchmark.pop('num_samples_validation', None)
    experiment_benchmark.pop('num_samples_test', None)

    # TODO @Johanna:
    # there might be a bug in the way we save the learning rate...
    # because we only save the benchmark after the training is done and the
    # as_dict function takes the learning rate directly from the optimizer instance,
    # this might not be the initial learning rate. 
    # Therefore it might be a better way to extract the initial learning rate from the
    # learning rate "metric", which was originally intented to visualize learning rate decay 
    # and such in tensorboard.
    
    # comment:
    # doesnt really matter because we dont need the initial learning rate
    # we compile the model such that we can keep training the model and it
    # only makes sense to continue with the same learning rate

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

    posw = tensorflow.constant(experiment_benchmark['positive_weights'], dtype=tensorflow.float32)
    negw = tensorflow.constant(experiment_benchmark['negative_weights'], dtype=tensorflow.float32)
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
        model = SimpleBaseArchitecture(DenseNet121, len(experiment_benchmark["label_columns"]))
    elif "DenseNet169" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.densenet.preprocess_input
        model = SimpleBaseArchitecture(DenseNet169, len(experiment_benchmark["label_columns"]))
    elif "InceptionV3" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.inception_v3.preprocess_input
        model = SimpleBaseArchitecture(InceptionV3, len(experiment_benchmark["label_columns"]))
    elif "ResNet101V2" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.resnet_v2.preprocess_input
        model = SimpleBaseArchitecture(ResNet101V2, len(experiment_benchmark["label_columns"]))
    elif "InceptionResNetV2" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.inception_resnet_v2.preprocess_input
        model = SimpleBaseArchitecture(InceptionResNetV2, len(experiment_benchmark["label_columns"]))
    elif "Xception" in experiment['name']:
        experiment_benchmark['preprocess_input_fn'] = tensorflow.keras.applications.xception.preprocess_input
        model = SimpleBaseArchitecture(Xception, len(experiment_benchmark["label_columns"]))
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