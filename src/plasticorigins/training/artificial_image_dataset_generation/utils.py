import cv2
import os
import numpy as np
import random
import hashlib


def get_background_names(folder_path):
    # Get a list of files in the folder
    file_list = os.listdir(folder_path)
    # Print the list of files
    return file_list


def generate_image_identifier(image, image_path):
    # we get a unique identifier using the resulting image
    image_data = cv2.imencode('.jpg', image)[1].tobytes()
    hash_object = hashlib.md5(image_data)
    identifier = hash_object.hexdigest()

    # get the batch number and image number to create a unique image identifier
    batch_num = image_path.split('/')[-2].split('_')[-1]
    img_num = image_path.split('/')[-1].split('.')[0]
    img_id = batch_num + '_' + img_num + '_' + identifier
    return img_id


def get_resize_ratio(per_w, per_h, box, img_size):
    min_x, min_y, width, height = box
    W, H = img_size
    return min(per_w * W / width, per_h * H / height)


def get_scale(bounding_box, img_size):
    max_per_w, max_per_h = 0.15, 0.15
    min_per_w, min_per_h = 0.05, 0.05
    max_per = get_resize_ratio(max_per_w, max_per_h, bounding_box, img_size)
    min_per = get_resize_ratio(min_per_w, min_per_h, bounding_box, img_size)
    scale = round(random.uniform(min_per, max_per), 2)
    return scale


def locations(img_size, bbox):
    min_x, min_y, width, height = bbox
    # The percentage of the shape width or height that can be cropped
    p = 0.25
    per_x = round(p*width)
    x = random.randint(- min_x - per_x, img_size[0] - min_x - width + per_x)

    # The shape should be around the horizon (the middle of the image) -->  positioned between 0.25*img_size[1] and 0.75*img_size[1]
    hor = round(0.3*img_size[1])
    y = random.randint(hor - min_y, img_size[1] - hor - min_y - height)
    return x, y


def paste_shape(resized_shape, canvas):
    transparence = random.uniform(0, 0.2)
    height, width, _ = resized_shape.shape
    # Calculate the alpha blend
    non_zero_pixels = np.any(resized_shape != 0, axis=2)
    np_condition = np.repeat(
        non_zero_pixels.reshape((height, width, 1)), 3, axis=2)

    alpha_blend = np.where(np_condition,
                           transparence * canvas + (1 - transparence) *
                           resized_shape,
                           canvas
                           )
    return alpha_blend


def get_bbox_from_polygon(polygon, img_size):
    # return the bounding box with this format: [center_x, center_y, width, height]
    contour = polygon.reshape((-1, 1, 2)).astype(np.int32)
    return get_bbox_from_contour(contour, img_size)


def get_bbox_from_contour(contour, img_size):
    # return the bounding box with this format: [center_x, center_y, width, height]
    x, y, width, height = cv2.boundingRect(contour)
    W, H = img_size
    x = max(x, 0)
    y = max(y, 0)
    x = min(x, W - width)
    y = min(y, H - height)

    center_x = x + width // 2
    center_y = y + height // 2
    return [center_x, center_y, width, height]


def crop_polygon(polygon, img_size):
    w, h = img_size
    # Create a mask for points outside the image boundaries
    mask = (polygon[:, 0] >= 0) & (polygon[:, 0] < w) & (
        polygon[:, 1] >= 0) & (polygon[:, 1] < h)

    # Filter the polygon points using the mask
    polygon = polygon[mask]
    return polygon


def rotate_img(image, polygon):
    # Get image height and width
    H, W = image.shape[:2]
    img_size = (W, H)
    # Define rotation center as image center
    center = (W / 2, H / 2)

    # Define the rotation angle in degrees
    angle = np.random.randint(-45, 45)

    # Define rotation matrix using cv2.getRotationMatrix2D
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # Apply rotation transformation using cv2.warpAffine
    rotated_image = cv2.warpAffine(image, rotation_matrix, img_size)
    # Apply same transformation to polygon using cv2.transform
    rotated_polygon = cv2.transform(
        np.array([polygon]), rotation_matrix)[0]

    rotated_polygon = crop_polygon(rotated_polygon, img_size)
    return rotated_image, rotated_polygon


def scale_img(image, polygon):
    # Get image height and width
    H, W = image.shape[:2]
    img_size = (W, H)
    center = (W/2, H/2)

    bounding_box = get_bbox_from_polygon(polygon, img_size)
    scale = get_scale(bounding_box, img_size)

    # Define scaling matrix using cv2.getRotationMatrix2D
    scaling_matrix = cv2.getRotationMatrix2D(center, 0, scale)
    # Apply scaling transformation using cv2.warpAffine
    scaled_image = cv2.warpAffine(image, scaling_matrix, img_size)
    # Apply same transformation to polygon
    scaled_polygon = np.round((polygon - center) * scale + center).astype(int)
    scaled_polygon = crop_polygon(scaled_polygon, img_size)
    return scaled_image, scaled_polygon


def center_polygon(image, polygon):
    H, W = image.shape[:2]
    img_size = (W, H)
    bounding_box = get_bbox_from_polygon(polygon, img_size)

    x, y, w, h = bounding_box
    # translate the image in order to have the object in the center
    # the new (x,y) coordinates of the shape should be (W/2, H/2)
    tx, ty = W/2 - x, H/2 - y

    # Apply scaling transformation using cv2.warpAffine
    shifted_image = cv2.warpAffine(image, np.float32(
        [[1, 0, tx], [0, 1, ty]]), img_size)
    # Apply same transformation to polygon
    shifted_polygon = polygon + [tx, ty]

    shifted_polygon = crop_polygon(shifted_polygon, img_size)
    return shifted_image, shifted_polygon


def shift_img(image, polygon):
    H, W = image.shape[:2]
    img_size = (W, H)

    bounding_box = get_bbox_from_polygon(polygon, img_size)
    tx, ty = locations(img_size, bounding_box)

    # Apply scaling transformation using cv2.warpAffine
    shifted_image = cv2.warpAffine(image, np.float32(
        [[1, 0, tx], [0, 1, ty]]), img_size)
    # Apply same transformation to polygon
    shifted_polygon = polygon + [tx, ty]

    shifted_polygon = crop_polygon(shifted_polygon, img_size)
    return shifted_image, shifted_polygon


def transform_img(image, polygon):
    # Apply rotation, scale, and shift to the image
    try:
        # first, we center the shape.
        image, polygon = center_polygon(image, polygon)

        # then we scale the image so that the shape will have a convenient size
        image, polygon = scale_img(image, polygon)

        # we randomly rotate the shape and image
        image, polygon = rotate_img(image, polygon)

        # we shift the image so that the shape will be placed in a random position in the generated image
        image, polygon = shift_img(image, polygon)

        return image, polygon
    except Exception as e:
        print(
            "An error occured while transforming the image: ", str(e))
