import datetime
import os
import os.path as op


def create_unique_folder(base_folder, filename):
    """Creates a unique folder based on the filename and timestamp"""
    folder_name = op.splitext(op.basename(filename))[0] + "_out_"
    folder_name += datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_dir = op.join(base_folder, folder_name)
    if not op.isdir(output_dir):
        os.mkdir(output_dir)
    return output_dir
