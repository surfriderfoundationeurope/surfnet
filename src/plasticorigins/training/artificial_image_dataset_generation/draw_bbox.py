import argparse
import cv2


def main(image_path, annotation_path):
    """Visualizes a bounding box annotation on an image.

    Args:
        image_path (str): Path to the image to visualize.
        annotation_path (str): Path to the annotation file containing class ID and bounding box coordinates.
    """

    image = cv2.imread(image_path)
    H, W = image.shape[:2]
    with open(annotation_path, 'r') as f:
        txt = f.read()
    class_id, x, y, w, h = txt.split(' ')

    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    x_min = int((x - w / 2) * W)
    y_min = int((y - h / 2) * H)
    x_max = int((x + w / 2) * W)
    y_max = int((y + h / 2) * H)

    cv2.rectangle(image, (x_min, y_min),
                  (x_max, y_max), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Contours with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--image_path", type=str, help="Path to the image")
    parser.add_argument("--annotation_path", type=str,
                        help="Path to the annotation")
    args = parser.parse_args()

    image_path = args.image_path
    annotation_path = args.annotation_path

    # Call the main function and pass the parameters
    main(image_path, annotation_path)
