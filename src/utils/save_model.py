import uuid
import json
from pathlib import Path
import os


def save_model(model, history, name, filename, description='', version="1"):
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

    # save model
    folderpath = basepath / 'models' / name
    path = folderpath / filename
    # make sure path exists, ceate one if necessary
    Path(folderpath).mkdir(parents=True, exist_ok=True)
    model.save(path)

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
        'validated': False,
        'test': None,
        'classification_report': None,
    }

    # append model data to log file
    log_file = basepath / 'logs/experiment-log.json'
    f = open(log_file, 'r')
    data = json.load(f)
    data['experiments'].append(log)
    f.close()

    f = open(log_file, 'w')
    json_data = json.dumps(data, indent=4)
    f.write(json_data)
    f.close()

    # reset workdir
    os.chdir(CURRENT_WORKING_DIR)

    return identifier


def model_set(identifier, attribute, value):
    """
    util function to set attributes in the log of the model

    Parameters:
    identifier: the id of the model
    attribute: the attribute name one wants to set
    value: the value that is set

    Returns:
    id string: the id of the model
    """

    CURRENT_WORKING_DIR = os.getcwd()
    # path main directory
    basepath = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    # set workdir to main directory
    os.chdir(basepath)

    # append model data to log file
    log_file = basepath / 'logs/experiment-log.json'
    f = open(log_file, 'r')
    data = json.load(f)
    for model in data['experiments']:
        if model['id'] == identifier:
            model[attribute] = value
    f.close()

    f = open(log_file, 'w')
    json_data = json.dumps(data, indent=4)
    f.write(json_data)
    f.close()

    # reset workdir
    os.chdir(CURRENT_WORKING_DIR)

    return identifier
