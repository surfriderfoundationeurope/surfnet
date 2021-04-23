import json 

with open('dataset.json','r') as surfrider_annotations_file:
    surfrider_annotations = [json.loads(line) for line in surfrider_annotations_file]

cat_name_to_cat_id = {'__background__':0,
                      'fragments':1,
                      'bottles':1,
                      'others':1}


coco_info_train = {'year':2020,
             'version':'0',
             'description':'A subset of images from Surfrider data for training',
             'contributor':'Surfrider',
             'url':'',
             'date_created':''}

coco_info_val = {'year':2020,
             'version':'0',
             'description':'A subset of images from Surfrider data for validation',
             'contributor':'Surfrider',
             'url':'',
             'date_created':''}

coco_license = {'id':0,'name':'','url':''}

coco_categories = [{'id':0,'name':'__background__','supercategory':'unknown'},
                   {'id':1,'name':'trash','supercategory':'unknown'}]

coco_images_train, coco_images_val = list(), list()
coco_annotations_train, coco_annotations_val = list(), list()

cnt_ann = 0

for ann_nb, surfrider_annotation in enumerate(surfrider_annotations):

    image_id = surfrider_annotation['id_file']
    width = surfrider_annotation['size']['width']
    height = surfrider_annotation['size']['height']
    file_name = surfrider_annotation['md5']

    if ann_nb < 24:
        coco_images_val.append({'id':int(image_id),
                            'width':int(width),
                            'height':int(height),
                            'file_name':file_name,
                            'license':0,
                            'flickr_url':'',
                            'coco_url':'',
                            'date_captured':''})
    else: 
        coco_images_train.append({'id':int(image_id),
                            'width':int(width),
                            'height':int(height),
                            'file_name':file_name,
                            'license':0,
                            'flickr_url':'',
                            'coco_url':'',
                            'date_captured':''})        


    for i,label in enumerate(surfrider_annotation['labels']):
        category_id = cat_name_to_cat_id[label['label']]

        bbox_surfrider = [int(dim) for dim in label['bbox']]
        bbox = [bbox_surfrider[0],
                     bbox_surfrider[1],
                     bbox_surfrider[2]-bbox_surfrider[0],
                     bbox_surfrider[3]-bbox_surfrider[1]]

        top_left_x, top_left_y, width, height = bbox
        top_right_x, top_right_y = top_left_x + width, top_left_y
        bottom_left_x, bottom_left_y = top_left_x, top_left_y + height
        bottom_right_x, bottom_right_y = top_left_x + width, top_left_y + height

        segmentation = [(top_left_x, top_left_y,
                         bottom_left_x, bottom_left_y,
                         bottom_right_x, bottom_right_y,
                         top_right_x, top_right_y)]



        area = bbox[2]*bbox[3]

        if ann_nb < 24: 
            coco_annotations_val.append({'id':cnt_ann,
                                    'image_id':int(image_id),
                                    'category_id':category_id,
                                    'segmentation':segmentation,
                                    #'area':area,
                                    'bbox':bbox,
                                    'iscrowd':0})
        else: 
            coco_annotations_train.append({'id':cnt_ann,
                                    'image_id':int(image_id),
                                    'category_id':category_id,
                                    'segmentation':segmentation,
                                    #'area':area,
                                    'bbox':bbox,
                                    'iscrowd':0})
        cnt_ann+=1


coco_train = {'info':coco_info_train,
        'images':coco_images_train,
        'annotations':coco_annotations_train,
        'categories':coco_categories,
        'license':coco_license}

coco_val = {'info':coco_info_val,
        'images':coco_images_val,
        'annotations':coco_annotations_val,
        'categories':coco_categories,
        'license':coco_license}


with open('instances_train.json','w') as coco_annotations_file:
    json.dump(coco_train,coco_annotations_file)



with open('instances_val.json','w') as coco_annotations_file:
    json.dump(coco_val,coco_annotations_file)





