from tensorflow.keras.applications.densenet import DenseNet121

from src.architectures.benchmarks.benchmark_definitions import benchmark_from_logs, simple_architecture_experiment
from src.utils.save_model import get_experiment, load_model
from src.architectures.simple.simple_base import SimpleBaseArchitecture
from src.utils.crm import CRM

def build_crm(model_name):
    exp_dict = get_experiment(name=model_name, version='1')
    benchmark = benchmark_from_logs(exp_dict['benchmark'])

    if 'DenseNet121' in model_name:
        architecture = DenseNet121

    num_classes = len(benchmark.label_columns)

    model = SimpleBaseArchitecture(
        architecture, num_classes, train_last_layer_only=False)
    path = str(load_model(name=model_name, version='1'))
    model.load_weights(path)

    return CRM(model, benchmark.label_columns, dims=(256, 256))
