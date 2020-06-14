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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Or 2, 3, etc. other than 0

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

def simple_architecture_experiment(benchmark, base_model_fn, classes ):
    model = SimpleBaseArchitecture(base_model_fn, len(classes))
    experiment = Experiment(benchmark, model)
    return experiment


architectures = {
    "InceptionV3": {
        "preprocess_input_fn":tf.keras.applications.inception_v3.preprocess_input,
        "model_fn": InceptionV3
    },
    "DenseNet121": {
        "preprocess_input_fn":tf.keras.applications.densenet.preprocess_input,
        "model_fn": DenseNet121
    },
}

"""
"InceptionResNetV2": {
    "preprocess_input_fn":tf.keras.applications.inception_resnet_v2.preprocess_input,
    "model_fn": InceptionResNetV2
},
"Xception": {
    "preprocess_input_fn":tf.keras.applications.xception.preprocess_input,
    "model_fn": Xception
},
"NASNetLarge": {
    "preprocess_input_fn":tf.keras.applications.nasnet.preprocess_input,
    "model_fn": NASNetLarge
}
"""
    
loss_functions = ["WBCE", "CWBCE"]
crop_confs = ["C1"]
for architecture_name, architecture in architectures.items():
    chexpert_benchmarks, _ = generate_benchmarks(batch_sizes = {"b": 32}, epoch_sizes = {"e": 20}, preprocess_input_fn = architecture["preprocess_input_fn"], split_seed = 6122156)
    for loss_function in loss_functions:
        for crop_conf in crop_confs:
            benchmark_key = loss_function+"_E20_B32_" + crop_conf
            chexpert_exp = simple_architecture_experiment(chexpert_benchmarks[benchmark_key], architecture["model_fn"], CHEXPERT_COLUMNS)
            print("START TRAINING FOR", chexpert_exp.model_name)
            chexpert_exp.run()