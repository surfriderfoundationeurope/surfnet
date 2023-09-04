from pathlib import Path
import argparse
from argparse import Namespace
import os
from tqdm import tqdm


#############
# remove  by excuting :
# python src/plasticorigins/training/data/remove_img.py --artificial-data /datadrive/data/artificial_data/
#############


def main(args: Namespace) -> None:
    """Main Function to remove data augmentation from Artificial Data.

    Args:
        args (argparse): list of arguments to build dataset for label mapping and training
    """

    # use data augmentation for artificial data only if original data have been processed
    artificial_data_dir = Path(args.artificial_data)
    artificial_train_files = [(artificial_data_dir / "images" / path).as_posix() for path in os.listdir(artificial_data_dir / "images")]
    original_data = [(artificial_data_dir / "images" / path).as_posix() for path in os.listdir(artificial_data_dir / "images") if len(path.split('/')[-1].split('_')) < 3]
    data_to_remove = list(set(artificial_train_files) - set(original_data))

    # remove data augmentation files
    for file in tqdm(data_to_remove):
        os.remove(file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build dataset")
    parser.add_argument("--artificial-data", type=str, help="path to artificial data folder")

    args = parser.parse_args()
    main(args)
