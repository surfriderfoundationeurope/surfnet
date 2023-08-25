import cv2
import numpy as np
import json
import argparse


def main(img_path, ann_path):
    """Visualizes image annotations by drawing object contours on the image.

    Args:
        img_path (str): Path to the image to visualize.
        ann_path (str): Path to the corresponding annotation file in JSON format.
    """

    # Load the image
    image = cv2.imread(img_path)

    with open(ann_path, 'r') as f:
        dataset = json.loads(f.read())

    file_name = image_path[image_path.index('batch'):]
    img_id = [i['id'] for i in dataset['images']
              if i['file_name'].lower() == file_name]
    img_id = img_id[0]
    annotations = [i for i in dataset['annotations']
                   if i['image_id'] == img_id]

    print(len(annotations))

    for ann in annotations:
        seg = ann['segmentation']
        bbox = ann['bbox']
        print(bbox, ann['id'])
        min_x, min_y, width, height = bbox
        min_x = int(min_x)
        min_y = int(min_y)
        width = int(width)
        height = int(height)

        # Extract polygon coordinates from annotation data
        polygon = np.array(np.array(seg), np.int32)
        polygon = polygon.reshape((-1, 2))

        # Draw the contour on the original image
        contour = polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Contours', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path to the image")
    parser.add_argument("--annotation_path", type=str,
                        help="Path to the annotation")
    args = parser.parse_args()

    image_path = args.image_path
    annotation_path = args.annotation_path

    # Call the main function and pass the parameters
    main(image_path, annotation_path)
