import uuid
import json
from pathlib import Path
import os
import tensorflow
from tensorflow.keras.models import load_model as load
from src.utils.storage import upload_file, download_file


def save_model(model, history, name, filename, description, version='1', upload=True):
    """
    Wrapper for the model.save function which logs the results

    Parameters:
    model (keras.model)
    history: the train history which is returned from the fit function
    name: the name of the model, defines also the folder where the model is saved
            (can be the same over different versions)
    filename: the filename of the model
    description: a description of the model
    version: the version of the model
    upload: whether the model should be uploaded to the gcp

    Returns:
    id string: the id of the model
    """

    if not model or not history:
        raise Exception('Hisory or model are not defined')

    CURRENT_WORKING_DIR = os.getcwd()
    # path main directory
    basepath = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    # set workdir to main directory
    os.chdir(basepath)

    # transform hisory values from np.float32 to regular floats
    for key in history.keys():
        history[key] = [float(val) for val in history[key]]

    identifier = str(uuid.uuid1())
    log = {
        'id': identifier,
        'name': name,
        'filename': filename,
        'version': version,
        'history': history,
        'description': description,
        # the following properties should be set using the `model_set` function
        # after eveluating the model. See function for documentation.
        'test': None,
        'classification_report': None,
    }

    # append model data to log file
    log_file = basepath / 'logs/unvalidated-experiment-log.json'
    with open(log_file, 'r') as f:
        data = json.load(f)

    # check for exisiting model with same name and version
    for experiment in data['experiments']:
        if experiment['name'] == log['name'] and experiment['version'] == log['version']:
            raise Exception(
                'There is already a model with the same name and version')

    data['experiments'].append(log)

    with open(log_file, 'w') as f:
        json_data = json.dumps(data, indent=4)
        f.write(json_data)

    # save model
    folderpath = basepath / 'models' / name
    path = folderpath / filename
    # make sure path exists, ceate one if necessary
    Path(folderpath).mkdir(parents=True, exist_ok=True)
    model.save(path)

    # upload model to gcp
    if upload:
        remote_name = log['id'] + '.h5'
        upload_file(path, remote_name)

    # reset workdir
    os.chdir(CURRENT_WORKING_DIR)

    return identifier


def model_set(identifier, attribute, value):
    """
    Util function to set attributes in the log of the model

    Valid attributes:
     test: List containing the test score and accuracy. Generated using the
           `model.evaluate_generator function
     classification_report: generated using `sklearn.metrics.classification_report`


    Parameters:
    identifier: the id of the model
    attribute: the attribute name one wants to set
    value: the value that is set

    Returns:
    id string: the id of the model
    """

    CURRENT_WORKING_DIR = os.getcwd()
    # path to main directory
    basepath = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    # set workdir to main directory
    os.chdir(basepath)

    # append model data to log file
    log_file = basepath / 'logs/unvalidated-experiment-log.json'
    with open(log_file, 'r') as f:
        data = json.load(f)
        for model in data['experiments']:
            if model['id'] == identifier:
                model[attribute] = value

    with open(log_file, 'w') as f:
        json_data = json.dumps(data, indent=4)
        f.write(json_data)

    # reset workdir
    os.chdir(CURRENT_WORKING_DIR)

    return identifier


def load_model(identifier=None, name=None, version=None):
    """
     Loads a given model (by identifier or by name and version) from gcp-storage if its not
     loaded locally

     Parameters:
     identifier: the id of the model
     name: the name of the model
     version: the version of the model

     Returns:
     keras model
    """

    if not (identifier or (name and version)):
        raise Exception(
            'You must specify the id, or the name and version of the model')

    CURRENT_WORKING_DIR = os.getcwd()
    # path main directory
    basepath = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    # set workdir to main directory
    os.chdir(basepath)

    # load logfile
    log_file = basepath / 'logs/experiment-log.json'
    with open(log_file, 'r') as f:
        data = json.load(f)
    experiments = data['experiments']

    # append unvalidated experiments
    unvalidated_log_file = basepath / 'logs/unvalidated-experiment-log.json'
    with open(unvalidated_log_file, 'r') as f:
        experiments = experiments + json.load(f)['experiments']

    # reset workdir
    os.chdir(CURRENT_WORKING_DIR)

    experiment = None
    for exp in experiments:
        if exp['id'] == identifier or (exp['name'] == name and exp['version'] == version):
            experiment = exp

    if not experiment:
        raise Exception('Model was not found')

    # build model path
    folderpath = basepath / 'models' / experiment['name']
    exp_path = folderpath / experiment['filename']

    # download model if it does not exist
    if not os.path.isfile(exp_path):
        bucket_filename = experiment['id'] + '.h5'
        Path(folderpath).mkdir(parents=True, exist_ok=True)
        download_file(bucket_filename, exp_path)

    return load(exp_path)
