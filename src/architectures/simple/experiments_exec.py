import os 
import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import traceback
import sys


basepath = Path(os.getcwd())
# make sure your working directory is the repository root.
if basepath.name != "idp-radio-1":
    os.chdir(basepath.parent.parent.parent)
load_dotenv(find_dotenv())

print(os.getcwd())

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# Run this before loading other dependencies, otherwise they might occupy memory on gpu 0 by default and it will stay that way

# Specify which GPU(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

config = tf.compat.v1.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.compat.v1.Session(config=config)

        
from tensorflow.keras.applications import InceptionV3, Xception, DenseNet121, InceptionResNetV2, ResNet152V2, NASNetLarge
from src.architectures.simple.simple_base import SimpleBaseArchitecture
from src.architectures.adv.guendel19 import densenet

from src.architectures.benchmarks.benchmark import Benchmark, Experiment
from src.architectures.benchmarks.benchmark_definitions import generate_benchmarks,simple_architecture_experiment, CHEXPERT_COLUMNS
from src.metrics.metrics import F2Score
from src.metrics.losses import WeightedBinaryCrossentropy, compute_class_weight
run_configs = [
    {
        "architectures" : {
            "DenseNet121": {
                "preprocess_input_fn":tf.keras.applications.densenet.preprocess_input,
                "model_fn": DenseNet121
            },
        },
        "columns": ['Cardiomegaly',
                    'Edema',
                    'Consolidation',
                    'Atelectasis',
                    'Pleural Effusion',
                    ],
        "epochs": 5,
        "batch_sizes": 32,
        "nan_replacement": 0,
        "dim":(256, 256),
        "optim": Adam(learning_rate=0.0001), # Adam()
        "split_valid_size": 0.1, 
        "name_suffix": "_D256_DS9010_LR4",
        "loss_functions": ["BCE"],
        "crop_confs":  ["C0"]
    }
]


"""

    {
        "architectures" : {
            "DenseNet121": {
                "preprocess_input_fn":tf.keras.applications.densenet.preprocess_input,
                "model_fn": DenseNet121
            },
        },
        "columns": ['Enlarged Cardiomediastinum',
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
                    'Fracture'],
        "epochs": 9,
        "batch_sizes": 32,
        "nan_replacement": 0,
        "name_suffix": "_D320",
        "loss_functions": ["CWBCE"],
        "crop_confs":  ["C0"]
    },
    {
        "architectures" : {
            "DenseNet121": {
                "preprocess_input_fn":tf.keras.applications.densenet.preprocess_input,
                "model_fn": DenseNet121
            },
        },
        "columns": ['Enlarged Cardiomediastinum',
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
                    'Fracture'],
        "epochs": 9,
        "batch_sizes": 32,
        "nan_replacement": 0,
        "name_suffix": "_D320",
        "loss_functions": ["BCE"],
        "crop_confs":  ["C0"]
    },
    

"InceptionResNetV2": {
    "preprocess_input_fn":tf.keras.applications.inception_resnet_v2.preprocess_input,
    "model_fn": InceptionResNetV2
},
"NASNetLarge": {
    "preprocess_input_fn":tf.keras.applications.nasnet.preprocess_input,
    "model_fn": NASNetLarge
}

    "Xception": {
        "preprocess_input_fn":tf.keras.applications.xception.preprocess_input,
        "model_fn": Xception
    },   
    
"""
estim_run_time = 0
for run_conf in run_configs:
    architectures = run_conf["architectures"]
    loss_functions = run_conf["loss_functions"]
    crop_confs = run_conf["crop_confs"]
    epoch_sizes = run_conf["epochs"]
    train_last_layer_only = run_conf["train_last_layer_only"] if "train_last_layer_only" in run_conf.keys() else False
    for architecture_name, architecture in architectures.items():
        for loss_function in loss_functions:
            for crop_conf in crop_confs:
                try:
                    chexpert_benchmarks, _ = generate_benchmarks(
                                                    # path = Path(os.environ.get("CHEXPERT_FULL_PREPROCESSED_DATASET_DIRECTORY")),
                                                     path = Path(os.environ.get("CHEXPERT_DATASET_DIRECTORY")),
                                                     name_suffix=run_conf["name_suffix"],
                                                     classes=run_conf["columns"],
                                                     train_labels = "nofinding_train.csv",
                                                     test_labels = "test.csv",
                                                     nan_replacement = run_conf["nan_replacement"], #float("NaN"),
                                                     batch_sizes = {"b": run_conf["batch_sizes"]},
                                                     epoch_sizes = {"e": epoch_sizes},
                                                     dim=run_conf["dim"],
                                                     optimizer = run_conf["optim"],
                                                     #crop = {"C1": False},
                                                     #crop = {"C0": False},
                                                     split_seed = 6122156,
                                                     split_valid_size = run_conf["split_valid_size"],
                                                     preprocess_input_fn = architecture["preprocess_input_fn"])
                    
                    benchmark_key = list(filter(lambda k: "_"+loss_function in "_"+k and crop_conf in k, list(chexpert_benchmarks.keys())))
                    if len(benchmark_key) > 0:
                        benchmark_key = benchmark_key[0]
                        print("Found benchmark {benchmark} for crop {crop_conf} and loss function {loss_function}".format(benchmark=benchmark_key,
                                                                                                                          crop_conf = crop_conf,
                                                                                                                          loss_function=loss_function))
                        print("Benchmark generator class number: ", len(chexpert_benchmarks[benchmark_key].label_columns))
                        print("Run Config class number: ", len(run_conf["columns"]))
                        chexpert_exp = simple_architecture_experiment(chexpert_benchmarks[benchmark_key], architecture["model_fn"], run_conf["columns"], train_last_layer_only=train_last_layer_only)
                        print("START TRAINING FOR", chexpert_exp.model_name)
                        
                        print(chexpert_exp.benchmark.as_dict())
                        
                        #print(chexpert_exp.model.summary())
                        
                        estim_run_time += epoch_sizes * 15
                        print ("Updated estim_run_time: ", estim_run_time / 60, " hours")
                        print()
                        chexpert_exp.run()
                    else:
                        print("Warning! Could not find benchmark for crop {crop_conf} and loss function {loss_function}".format(crop_conf = crop_conf,
                                                                                                                                loss_function=loss_function))
                except Exception as err:
                    print ("Experiment failed...")
                    print(err)
                    print(traceback.format_exc())
                    print(sys.exc_info()[2])
                    

print ("Final estim_run_time: ", estim_run_time / 60, " hours")
