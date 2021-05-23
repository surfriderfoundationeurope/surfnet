import os
import json 

def main(args):
    annotations_grouped = {'categories':[],
                           'images':[],
                           'videos':[],
                           'annotations':[]}

    annotation_filenames = [filename for filename in sorted(os.listdir(args.input_dir)) if filename.endswith('.json')]

    nb_annotations_processed = 0
    nb_images_processed = 0

    for id, annotation_filename in enumerate(annotation_filenames):
        annotations_grouped['videos'].append({'file_name':annotation_filename.strip('.json') + '.MP4', 'id':id+1})

        with open(args.input_dir + annotation_filename,'r') as annotation_file:
            single_annotation = json.load(annotation_file)

        for category in single_annotation['categories']:
            if category not in annotations_grouped['categories']: annotations_grouped['categories'].append(category)
        
        for image in single_annotation['images']:
            image['video_id'] = id+1
            image['id'] = image['id'] + nb_images_processed
            annotations_grouped['images'].append(image)

        for annotation in single_annotation['annotations']:
            annotation['image_id'] = annotation['image_id'] + nb_images_processed
            annotation['id'] = annotation['id'] + nb_annotations_processed
            annotations_grouped['annotations'].append(annotation)

        nb_images_processed+=len(single_annotation['images'])
        nb_annotations_processed+=len(single_annotation['annotations'])

    with open(args.input_dir + 'annotations.json','w') as f:
        json.dump(annotations_grouped, f)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Surfnet training')

    parser.add_argument('--input_dir', type=str)

    args = parser.parse_args()

    main(args)