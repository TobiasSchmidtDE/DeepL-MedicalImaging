"""This file will be run when deploying the docker container"""

import os.path
import sys

# check if dataset has been downloaded
if not os.path.isdir(os.curdir + '/data/dev_dataset/CheXpert-v1.0-small'):
    sys.exit('Dataset not found')
