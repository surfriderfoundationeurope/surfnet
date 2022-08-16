"""The ``make_dataset`` submodule provides only one main function to get annotations, 
build annotations files for yolo and build dataset for training.

"""

from data_processing import get_annotations_from_files, get_annotations_from_db, find_img_ids_to_exclude
from data_processing import generate_yolo_files, get_train_valid, build_yolo_annotations_for_images
from pathlib import Path
import argparse


def main(args:argparse) -> None:

    """Main Function to get annotations, build annotations files for yolo and build dataset for training.

    Args:
        args (argparse): list of arguments to build dataset for label mapping and training
    """

    data_dir = Path(args.data_dir)
    image_dir = Path(args.images_dir)

    if args.bbox_filename and args.images_filename:
        df_bboxes, df_images = get_annotations_from_files(data_dir,
                                                        args.bbox_filename,
                                                        args.images_filename)
    elif args.password:
        print("getting annotations from db")
        df_bboxes, df_images = get_annotations_from_db(args.password)

    else:
        print("either a password must be set, or bbox and images filenames")
        return

    if args.exclude_img_folder:
        to_exclude = find_img_ids_to_exclude(args.exclude_img_folder)
    else:
        to_exclude = None

    yolo_filelist, cpos, cneg = build_yolo_annotations_for_images(data_dir, image_dir, args.bbox_filename,
            df_bboxes, df_images, args.limit_data, args.name_images_folder, args.name_labels_folder, to_exclude)

    print(f"found {cpos} valid annotations with images and {cneg} unmatched annotations")

    train_files, val_files = get_train_valid(yolo_filelist, args.split)

    generate_yolo_files(data_dir, train_files, val_files)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument('--data-dir', type=str, help="path to main data folder")
    parser.add_argument('--images-dir', type=str, help="path to image folder")
    parser.add_argument('--name-images-folder', type=str, default="images_test", help="name of processed images")
    parser.add_argument('--name-labels-folder', type=str, default="labels_test", help="name of processed labels")
    parser.add_argument('--password', type=str, help="password for connection to DB")
    parser.add_argument('--bbox-filename', type=str, default="")
    parser.add_argument('--images-filename', type=str, default="")
    parser.add_argument('--split', type=float, default=0.85)
    parser.add_argument('--limit-data', type=int, default=0)
    parser.add_argument('--exclude-img-folder', type=str, help="the path to the folder which contains images and annotations, from which we can find the img ids to exclude")
    args = parser.parse_args()

    main(args)
