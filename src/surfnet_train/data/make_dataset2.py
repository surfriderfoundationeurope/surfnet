
from data_processing import coco2yolo
from data_processing import get_df_train_val
from pathlib import Path
from sklearn.model_selection import KFold
import pandas as pd
import psycopg2




def __main__(args):
    data_dir = Path("../data/")
    if args.use_db:
        df_bboxes, df_images = get_annotations_from_db()
    else:
        df_bboxes, df_images = get_annotations_from_files(data_dir,
                                            args.bbox_filename,
                                            args.images_filename)

    yolo_filelist = build_yolo_annotations_for_images(data_dir, df_bboxes, df_images)

    df_train_valid = get_df_train_val(annotation_file, data_dir, df_images)

    kf = KFold(n_splits=7)
    df_train_valid = df_train_valid.reset_index(drop=True)
    df_train_valid['fold'] = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train_valid)):
        df_train_valid.loc[val_idx, 'fold'] = fold

    FOLD = 1

    train_files = []
    val_files   = []
    train_df = df_train_valid.query("fold!=@FOLD")
    valid_df = df_train_valid.query("fold==@FOLD")

    train_files = list(train_df["img_name"].unique())
    val_files   = list(valid_df["img_name"].unique())

    generate_data_files(data_dir, train_files, val_files)
