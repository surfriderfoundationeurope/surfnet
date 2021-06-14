import json
import psycopg2

# Update connection string information
host = "pgdb-plastico-prod.postgres.database.azure.com"
dbname = "plastico-prod"
user = "reader_user@pgdb-plastico-prod"
password = input('Enter password:')
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

conn.close()

annotations = []
image_name_conversion_table = []

for raw_annotation in raw_annotations:
	# date = raw_annotation[2]
	# if date.day == 14 and date.month == 6:
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
    image_name_conversion_table.append({
		"id" : raw_image_info[0],
		"id_creator_fk" : raw_image_info[1],
		"createdon" : raw_image_info[2],
		"filename" : raw_image_info[3],
		"view" : raw_image_info[4],
		"image_quality" : raw_image_info[5],
		"context" : raw_image_info[6],
		"container_url" : raw_image_info[7],
		"blob_name" : raw_image_info[8]
	})

images_id_refs = list(set([annotation['id_ref_images_for_labelling'] for annotation in annotations]))
image_dbid_to_cocoid = {image_dbid:image_cocoid for image_cocoid, image_dbid in enumerate(images_id_refs)}

image_idref_to_image_filename = {image['id']:image['filename'] for image in image_name_conversion_table}

coco_images = [{'id':image_cocoid, 'file_name':image_idref_to_image_filename[image_dbid]} for image_dbid, image_cocoid in image_dbid_to_cocoid.items()]

coco_categories = [{'id':0,'name':'__background__','supercategory':'unknown'},
                   {'id':1,'name':'trash','supercategory':'unknown'}]

coco_annotations = list()

for annotation_id, annotation in enumerate(annotations):
    image_dbid = annotation['id_ref_images_for_labelling']
    bbox = [annotation['location_x'], annotation['location_y'], annotation['width'], annotation['height']]

    coco_annotations.append({'id':annotation_id,
                             'image_id':image_dbid_to_cocoid[image_dbid],
                             'bbox':bbox,
                             'category_id':1})

coco = {'images':coco_images,'annotations':coco_annotations,'categories':coco_categories}

with open('data/images/annotations/instances_train_new.json','w') as f:
    json.dump(coco, f)


