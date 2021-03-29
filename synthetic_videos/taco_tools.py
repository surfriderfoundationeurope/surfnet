import cv2
import random
import json
import os
import numpy as np
taco_path = None

def load_TACO():

    anns_file_path = taco_path + 'annotations.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())
    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1

    print('Number of super categories:', nr_super_cats)
    print('Number of categories:', nr_cats)
    print('Number of annotations:', nr_annotations)
    print('Number of images:', nr_images)

    # map to our categories
    ids_plastic = list(range(36,42))
    ids_other = [10,11,12,43,44,45,46,47,51,53]
    ids_bottle = [4,5]

    dict_label_to_ann_ids = {"bottle":[], "fragment":[], "other":[]}
    for idx, ann in enumerate(anns):
        if ann["category_id"] in ids_bottle:
            dict_label_to_ann_ids["bottle"] += [idx]
        elif ann["category_id"] in ids_plastic:
            dict_label_to_ann_ids["fragment"] += [idx]
        elif ann["category_id"] in ids_other:
            dict_label_to_ann_ids["other"] += [idx]

    return anns, imgs, dict_label_to_ann_ids

def crop_img(img, pts):
    # print(pts.shape)
    # get bounding box
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    print(mask.shape)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    
    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the alpha channel
    rgba = cv2.cvtColor(dst, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask
    return rgba

def get_random_trash(label, anns, imgs, dict_label_to_ann_ids):
    list_idx = dict_label_to_ann_ids[label]
    idx = random.choice(list_idx)
    ann = anns[idx]
    img_id = ann['image_id']
    img_path = os.path.join(taco_path, imgs[img_id]['file_name'])

    img = cv2.imread(img_path)
    #idx_seg = random.choice(len(ann["segmentation"]))
    seg = random.choice(ann['segmentation'])
    pts = np.array(list(zip(seg[::2], seg[1::2]))).astype(int)

    if pts.shape[0] > 0 and pts.shape[1] > 0:
        return crop_img(img, pts)
    else: 
        return get_random_trash(label, anns, imgs, dict_label_to_ann_ids)
