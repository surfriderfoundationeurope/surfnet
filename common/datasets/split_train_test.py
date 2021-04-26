import json 

def main(args):
    train_proportion = 0.8
    with open(args.annotation_dir+'annotations_resized.json','r') as f:
        annotations_full = json.load(f)

    annotations_train = dict()
    annotations_val = dict()

    annotations_train['categories'] = annotations_full['categories']
    annotations_val['categories'] = annotations_full['categories']

    num_videos_train = int(0.8*len(annotations_full['videos']))
    annotations_train['videos'] = annotations_full['videos'][:num_videos_train]
    annotations_val['videos'] = annotations_full['videos'][num_videos_train:]

    video_ids_val = [video['id'] for video in annotations_val['videos']]

    annotations_train['images'] = []
    annotations_val['images'] = []

    for image in annotations_full['images']: 
        if image['video_id'] in video_ids_val:
            annotations_val['images'].append(image)
        else:
            annotations_train['images'].append(image)

    image_ids_val = [image['id'] for image in annotations_val['images']]


    annotations_train['annotations'] = []
    annotations_val['annotations'] = []

    for annotation in annotations_full['annotations']:
        if annotation['image_id'] in image_ids_val:
            annotations_val['annotations'].append(annotation)
        else:
            annotations_train['annotations'].append(annotation)

    with open(args.annotation_dir+'annotations_val.json','w') as f:
        json.dump(annotations_val,f)
    with open(args.annotation_dir+'annotations_train.json','w') as f: 
        json.dump(annotations_train,f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_dir', default='data/generated_videos/', type=str)
    args = parser.parse_args()
    main(args)