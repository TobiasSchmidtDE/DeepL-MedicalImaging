import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, Xception, DenseNet121, InceptionResNetV2, ResNet152V2, NASNetLarge, DenseNet169

from src.architectures.benchmarks.benchmark_definitions import benchmark_from_logs, simple_architecture_experiment
from src.utils.save_model import get_experiment, load_model
from src.architectures.simple.simple_base import SimpleBaseArchitecture
from src.utils.crm import CRM


def build_crm(model_name):
    # Author: Kristian
    """
        Utility function to build a CRM class using just the model name

        params:
            model_name (string)
        returns:
            crm (CRM)
    """

    exp_dict = get_experiment(name=model_name, version='1')
    benchmark = benchmark_from_logs(exp_dict['benchmark'])

    if 'DenseNet121' in model_name:
        architecture = DenseNet121
        preprocess_fn = tf.keras.applications.densenet.preprocess_input
    elif 'DenseNet169' in model_name:
        architecture = DenseNet169
        preprocess_fn = tf.keras.applications.densenet.preprocess_input
    elif 'InceptionV3' in model_name:
        architecture = InceptionV3
        preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
    elif 'Xception' in model_name:
        architecture = Xception
        preprocess_fn = tf.keras.applications.xception.preprocess_input
    elif 'InceptionResNetV2' in model_name:
        architecture = InceptionResNetV2
        preprocess_fn = tf.keras.applications.inception_resnet_v2.preprocess_input

    else:
        raise Exception('Architecture not defined in build_crm function')

    num_classes = len(benchmark.label_columns)

    model = SimpleBaseArchitecture(
        architecture, num_classes, train_last_layer_only=False)
    path = str(load_model(name=model_name, version='1'))
    model.load_weights(path)

    return CRM(model, benchmark.label_columns, dims=(256, 256), preprocess_fn=preprocess_fn)
