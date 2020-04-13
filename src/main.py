"""This file can be run in order to check if the environment is set up correctly"""

import os.path
import sys
import tensorflow as tf

# check if dataset has been downloaded
if not os.path.isdir(os.curdir + '/data/dev_dataset/CheXpert-v1.0-small'):
    sys.exit('Dataset not found')

# check tensorflow gpu installation
tf.print(tf.constant('Hello, TensorFlow!'))
