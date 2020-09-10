import os 
import datetime
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import tensorflow as tf

basepath = Path(os.getcwd())
# make sure your working directory is the repository root.
if basepath.name != "idp-radio-1":
    os.chdir(basepath.parent.parent.parent)
load_dotenv(find_dotenv())
basepath = Path(os.getcwd())


# Specify which GPU(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

#config = tf.compat.v1.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True, log_device_placement=True)
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 1.2

import numpy as np

import traceback
from sklearn.metrics import classification_report
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam, SGD
from keras.utils.generic_utils import get_custom_objects

from tensorflow.keras.applications import InceptionV3, Xception, DenseNet121, InceptionResNetV2, ResNet152V2, NASNetLarge
from src.architectures.simple.simple_base import SimpleBaseArchitecture
from src.architectures.simple.load_model import get_all_experiment_logs, get_experiment_from_logs, benchmark_from_logs, get_preprocessing_for_architecture, get_model_build_function, rebuild_experiment, difference_test_results
from src.architectures.benchmarks.benchmark import Benchmark, Experiment
from src.architectures.benchmarks.benchmark_definitions import generate_benchmarks,simple_architecture_experiment, Chexpert_Benchmark, CHEXPERT_COLUMNS, METRICS, SINGLE_CLASS_METRICS
from src.metrics.metrics import F2Score
from src.metrics.losses import WeightedBinaryCrossentropy, compute_class_weight
from src.utils.save_model import get_experiment, load_model
import math

experiments = get_all_experiment_logs()
experiments = [exp for exp in experiments if "Failed" not in exp["name"]]
experiments = [exp for exp in experiments if "N12" in exp["name"]]
experiments = [exp for exp in experiments if "num_samples_test" in exp["benchmark"].keys() ]
experiments = [exp for exp in experiments if exp["benchmark"]["num_samples_test"] == 234 ]
print("Num of exps:", len(experiments))

def build_model(model_name):
    exp_dict = get_experiment(name=model_name, version='1')
    benchmark = benchmark_from_logs(exp_dict)

    if 'DenseNet121' in model_name:
        architecture = DenseNet121
    elif 'DenseNet169' in model_name:
        architecture = DenseNet169
    elif 'InceptionV3' in model_name:
        architecture = InceptionV3
    elif 'Xception' in model_name:
        architecture = Xception
    elif 'InceptionResNetV2' in model_name:
        architecture = InceptionResNetV2
    else:
        raise Exception('Architecture not defined in build_crm function')


    num_classes = len(benchmark.label_columns)

    model = SimpleBaseArchitecture(
        architecture, num_classes, train_last_layer_only=False)
    path = str(load_model(name=model_name, version='1'))
    model.load_weights(path)
    
    model.compile(optimizer=benchmark.optimizer,
                           loss=benchmark.loss,
                           metrics=benchmark.metrics)

    return model, benchmark


def evaluate(model, benchmark):
    eval_res = model.evaluate(
            x=benchmark.testgen, steps=len(benchmark.testgen), verbose=1)

    metric_names = benchmark.as_dict()["metrics"]
    eval_metrics = dict(
        zip(["loss"] + metric_names, [float(i) for i in eval_res if not math.isnan(float(i))]))

    return  {
        "metrics": eval_metrics
    }

def ev1():
    exp_dict = experiments[1].copy()
    model, benchmark = build_model(exp_dict['name'])
    res = evaluate(model, benchmark)
    print(exp_dict["test"]["auc"] - res["metrics"]["auc"])


def ev2():
    test = experiments[2].copy()
    model2, benchmark2 = build_model(test['name'])
    res2 = evaluate(model2, benchmark2)
    print(test["test"]["auc"] - res2["metrics"]["auc"])
    
print("Evaluation 1")
ev1()

print("Evaluation 2")
ev2()