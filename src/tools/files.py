import os
import os.path as op
from urllib.request import urlretrieve
import datetime


def create_unique_folder(base_folder, filename):
    """Creates a unique folder based on the filename and timestamp
    """
    folder_name = op.splitext(op.basename(filename))[0] + "_out_"
    folder_name += datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    output_dir = op.join(base_folder, folder_name)
    if not op.isdir(output_dir):
        os.mkdir(output_dir)
    return output_dir


def download_model_from_url(url, filename, logger):
    """
    Download a model file and place it in the corresponding folder if it does
    not already exists
    """
    model_filename = op.realpath('./models/' + filename)
    if not op.exists(model_filename):
        logger.info('---Downloading model...')
        urlretrieve(url, model_filename)
    else:
        logger.info('---Model already downloaded.')
    return model_filename
