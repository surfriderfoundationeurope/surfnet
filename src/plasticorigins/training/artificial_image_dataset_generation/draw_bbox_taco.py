import argparse
import cv2
import json


def main(image_path, annotation_path):
    """Visualizes bounding box annotations on an image.

    Args:
        image_path (str): Path to the image to visualize.
        annotation_path (str): Path to the JSON annotation file containing bounding box information.
    """

    image = cv2.imread(image_path)
    H, W = image.shape[:2]

    with open(annotation_path, 'r') as f:
        dataset = json.loads(f.read())
    file_name = image_path[image_path.index('batch'):]

    img_id = [i['id'] for i in dataset['images']
              if i['file_name'].lower() == file_name]
    img_id = img_id[0]
    annotations = [i for i in dataset['annotations']
                   if i['image_id'] == img_id]

    for ann in annotations:
        bbox = ann['bbox']
        min_x, min_y, width, height = bbox
        print(bbox, ann['id'])
        min_x = int(min_x)
        min_y = int(min_y)
        width = int(width)
        height = int(height)

        # Draw the bounding box on the image
        cv2.rectangle(image, (min_x, min_y),
                      (min_x + width, min_y + height), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Contours with Bounding Box', image)
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
