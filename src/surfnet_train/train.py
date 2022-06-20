import yaml
from data_processing import coco2yolo
from data_processing import get_df_train_val
from pathlib import Path
from sklearn.model_selection import KFold
import pandas as pd
import psycopg2

def get_annotations_from_db():
    """ Gets the data from the database
    """
    # Update connection string information
    host = "pgdb-plastico-prod.postgres.database.azure.com"
    dbname = "plastico-prod"
    user = "reader_user@pgdb-plastico-prod"
    # password = input('Enter password:')
    password = 'SurfReader!'
    sslmode = "require"

    # Construct connection string
    conn_string = f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
    conn = psycopg2.connect(conn_string)
    print("Connection established")

    # Fetch all rows from table
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM "label".bounding_boxes')
    raw_annotations = cursor.fetchall()

    cursor.execute('SELECT * FROM "label".images_for_labelling')
    raw_images_info = cursor.fetchall()

    #cursor.execute('SELECT * FROM "campaign".trash_type')
    #raw_category_info = cursor.fetchall()
    df_bboxes = pd.DataFrame(raw_annotations,
                             columns=["id","id_creator_fk","createdon","id_ref_trash_type_fk",
                                      "id_ref_images_for_labelling","location_x","location_y",
                                      "width","height"])

    df_images = pd.DataFrame(raw_images_info,
                             columns=["id","id_creator_fk","createdon","filename","view",
                                      "image_quality","context","container_url","blob_name"])
    conn.close()
    return df_bboxes, df_images.set_index("id") #, raw_category_info


def get_annotations_from_files(input_dir, bbox_filename, images_filename):
    return pd.read_csv(input_dir / bbox_filename),
           pd.read_csv(input_dir / images_filename).set_index("id")
           #pd.read_csv(input_dir / trash_filename)


def save_annotations_to_files(output_dir, df_bboxes, df_images):
    df_bboxes.to_csv(output_dir / "bbox.csv")
    df_images.to_csv(output_dir / "images.csv")


def generate_yolo_files(output_dir, train_files, val_files):
    """ Generates data files for yolo training: train.txt, val.txt and data.yaml
    """
    output_dir = Path(output_dir)
    with open(output_dir / 'train.txt', 'w') as f:
        for path in train_files:
            f.write(path+'\n')

    with open(output_dir / 'val.txt', 'w') as f:
        for path in val_files:
            f.write(path+'\n')

    data = dict(
        path  = './../',
        train =  output_dir / 'train.txt' ,
        val   =  output_dir / 'val.txt'),
        nc    = 10,
        names = ['Sheet / tarp / plastic bag / fragment', 'Insulating material', 'Bottle-shaped', 'Can-shaped', 'Drum', 'Other packaging', 'Tire', 'Fishing net / cord', 'Easily namable', 'Unclear'],
        )

    with open(output_dir / 'data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

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
