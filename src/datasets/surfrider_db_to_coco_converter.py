import json
from collections import defaultdict
import psycopg2

# Update connection string information
host = "pgdb-plastico-prod.postgres.database.azure.com"
dbname = "plastico-prod"
user = "reader_user@pgdb-plastico-prod"
# password = input('Enter password:')
password = 'SurfReader!'
sslmode = "require"

# Construct connection string
conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
conn = psycopg2.connect(conn_string)
print("Connection established")

# Fetch all rows from table
cursor = conn.cursor()

cursor.execute('SELECT * FROM "label".bounding_boxes')
raw_annotations = cursor.fetchall()

cursor.execute('SELECT * FROM "label".images_for_labelling')
raw_images_info = cursor.fetchall()

cursor.execute('SELECT * FROM "campaign".trash_type')
raw_category_info = cursor.fetchall()

conn.close()

annotations = []
images_db_infos = dict()

for raw_annotation in raw_annotations:
	date = raw_annotation[2]
	if date.month >= 6 and date.day >= 4:
		annotations.append({
			"id" : raw_annotation[0],
			"id_creator_fk" : raw_annotation[1],
			"createdon" : raw_annotation[2],
			"id_ref_trash_type_fk" : raw_annotation[3],
			"id_ref_images_for_labelling" : raw_annotation[4],
			"location_x" : raw_annotation[5],
			"location_y" : raw_annotation[6],
			"width" : raw_annotation[7],
			"height" : raw_annotation[8]
		})

for raw_image_info in raw_images_info:
    images_db_infos[raw_image_info[0]] = {
		"id_creator_fk" : raw_image_info[1],
		"createdon" : raw_image_info[2],
		"filename" : raw_image_info[3],
		"view" : raw_image_info[4],
		"image_quality" : raw_image_info[5],
		"context" : raw_image_info[6],
		"container_url" : raw_image_info[7],
		"blob_name" : raw_image_info[8]}

image_db_id_to_image_filename = {k:v['filename'] for k,v in images_db_infos.items()}

images_db_ids = list(set([annotation['id_ref_images_for_labelling'] for annotation in annotations]))
image_filenames = list(set([image_db_id_to_image_filename[image_db_id] for image_db_id in images_db_ids]))
image_filename_to_image_coco_id = {image_filename:image_coco_id for image_coco_id, image_filename in enumerate(image_filenames)}

coco_images = [{'id':image_coco_id, 'file_name':image_filename} \
	for image_filename, image_coco_id in image_filename_to_image_coco_id.items()]

coco_categories = [{'id':0,'name':'__background__','supercategory':'unknown'}] \
                + [{'id':raw_category[0],'name':raw_category[1],'supercategory':'trash'} for raw_category in raw_category_info]

coco_annotations = list()

for annotation_id, annotation in enumerate(annotations):
    image_db_id = annotation['id_ref_images_for_labelling']
    coco_annotations.append({'id':annotation_id,
                             'image_id':image_filename_to_image_coco_id[image_db_id_to_image_filename[image_db_id]],
                             'bbox':[annotation['location_x'], annotation['location_y'], annotation['width'], annotation['height']],
                             'category_id':annotation['id_ref_trash_type_fk']})

coco = {'images':coco_images,'annotations':coco_annotations,'categories':coco_categories}

with open('data/images/annotations/instances_multiclass.json','w') as f:
    json.dump(coco, f)


