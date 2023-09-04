from pathlib import Path
import argparse
from argparse import Namespace
import os


#############
# COPY AND RENAME OLD TRAIN AND VAL TXT
# FILL NEW TRAIN TXT by excuting :
# python src/plasticorigins/training/data/DA_for_GenData.py --data-dir /datadrive/data/data_20062022 --artificial-data /datadrive/data/artificial_data
# RENAME WITH _DA SUFFIX AFTER
#############


def main(args: Namespace) -> None:
    """Main Function to write new images paths for training.

    Args:
        args (argparse): list of arguments to build dataset for label mapping and training
    """

    data_dir = Path(args.data_dir)

    # use data augmentation for artificial data only if original data have been processed
    artificial_data_dir = Path(args.artificial_data)
    artificial_train_files = [Path(path).as_posix() for path in os.listdir(artificial_data_dir / "images")]

    # concatenate original images and artificial data
    with open(data_dir / "train.txt", "w") as f:
        for path in artificial_train_files:
            f.write(path + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build dataset")
    parser.add_argument("--data-dir", type=str, help="path to main data folder")
    parser.add_argument("--artificial-data", type=str, help="path to artificial data folder")

    args = parser.parse_args()
    main(args)
