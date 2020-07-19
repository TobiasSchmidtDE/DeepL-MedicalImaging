import os 
import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

basepath = Path(os.getcwd())
# make sure your working directory is the repository root.
if basepath.name != "idp-radio-1":
    os.chdir(basepath.parent.parent.parent)
load_dotenv(find_dotenv())

print(os.getcwd())

import os 
import tensorflow as tf
from pathlib import Path

# Run this before loading other dependencies, otherwise they might occupy memory on gpu 0 by default and it will stay that way

# Specify which GPU(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

        
from tensorflow.keras.applications import InceptionV3, Xception, DenseNet121, InceptionResNetV2, ResNet152V2, NASNetLarge
from src.architectures.simple.simple_base import SimpleBaseArchitecture
from src.architectures.adv.guendel19 import densenet

from src.architectures.benchmarks.benchmark import Benchmark, Experiment
from src.architectures.benchmarks.benchmark_definitions import generate_benchmarks, CHEXPERT_COLUMNS
from src.metrics.metrics import F2Score
from src.metrics.losses import WeightedBinaryCrossentropy, compute_class_weight

def simple_architecture_experiment(benchmark, base_model_fn, classes, train_last_layer_only=False ):
    model = SimpleBaseArchitecture(base_model_fn, len(classes), train_last_layer_only = train_last_layer_only)
    experiment = Experiment(benchmark, model)
    return experiment

run_configs = [
   
    {
        "architectures" :{
            "InceptionV3": {
                "preprocess_input_fn":tf.keras.applications.inception_v3.preprocess_input,
                "model_fn": InceptionV3
            },
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
        "epochs": 4,
        "loss_functions": ["CWBCE", "WBCE"],
        "crop_confs":  ["C1"]
    },
    
]


"""


    {
        "architectures" : {
            "InceptionV3": {
                "preprocess_input_fn":tf.keras.applications.inception_v3.preprocess_input,
                "model_fn": InceptionV3
            },
            "DenseNet121": {
                "preprocess_input_fn":tf.keras.applications.densenet.preprocess_input,
                "model_fn": DenseNet121
            },
        },
        "columns": CHEXPERT_COLUMNS,
        "epochs": 3,
        "loss_functions": ["CWBCE"],
        "crop_confs":  ["C1"]
    },
{
        "architectures" : {
            "InceptionV3": {
                "preprocess_input_fn":tf.keras.applications.inception_v3.preprocess_input,
                "model_fn": InceptionV3
            },
            "DenseNet121": {
                "preprocess_input_fn":tf.keras.applications.densenet.preprocess_input,
                "model_fn": DenseNet121
            },
        },
        "columns": CHEXPERT_COLUMNS,
        "epochs": 5,
        "train_last_layer_only": True,
        "loss_functions": ["CWBCE", "WBCE"],
        "crop_confs":  ["C1"]
    },
 {
        "architectures" :{
            "InceptionV3": {
                "preprocess_input_fn":tf.keras.applications.inception_v3.preprocess_input,
                "model_fn": InceptionV3
            },
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
                    'Fracture',
                    'Support Devices'],
        "epochs": 5,
        "loss_functions": ["CWBCE", "WBCE"],
        "crop_confs":  ["C1"]
    },
    
    {
        "architectures" : {
            "InceptionV3": {
                "preprocess_input_fn":tf.keras.applications.inception_v3.preprocess_input,
                "model_fn": InceptionV3
            },
            "DenseNet121": {
                "preprocess_input_fn":tf.keras.applications.densenet.preprocess_input,
                "model_fn": DenseNet121
            },
        },
        "columns": ['No Finding',
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
                    'Fracture'],
        "epochs": 6,
        "loss_functions": ["CWBCE", "WBCE"],
        "crop_confs":  ["C1"]
    }

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
        chexpert_benchmarks, _ = generate_benchmarks(path = Path(os.environ.get("CHEXPERT_FULL_DATASET_DIRECTORY")),
                                                     classes=run_conf["columns"],
                                                     #train_labels = "nofinding_train.csv",
                                                     #nan_replacement = float("NaN"),
                                                     batch_sizes = {"b": 32},
                                                     epoch_sizes = {"e": epoch_sizes},
                                                     crop = {"C1": True},
                                                     split_seed = 6122156, 
                                                     preprocess_input_fn = architecture["preprocess_input_fn"])


        for loss_function in loss_functions:
            for crop_conf in crop_confs:
                try:
                    benchmark_key = list(filter(lambda k: "_"+loss_function in "_"+k and crop_conf in k, list(chexpert_benchmarks.keys())))
                    if len(benchmark_key) > 0:
                        benchmark_key = benchmark_key[0]
                        print("Found benchmark {benchmark} for crop {crop_conf} and loss function {loss_function}".format(benchmark=benchmark_key,
                                                                                                                          crop_conf = crop_conf,
                                                                                                                          loss_function=loss_function))
                        
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
                    

print ("Final estim_run_time: ", estim_run_time / 60, " hours")
