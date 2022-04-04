from matplotlib.pyplot import grid
import numpy as np
import math
import cv2
import torch
import torchvision.transforms.functional as F
from plasticorigins.detection.centernet.models import create_model as create_base

class ResizeForCenterNet(object):
    def __init__(self, fix_res=False):
        self.fix_res = fix_res

    def __call__(self, image):
        if self.fix_res:
            new_h = 512
            new_w = 512
        else:
            w, h = image.size
            new_h = (h | 31) + 1
            new_w = (w | 31) + 1
        image = F.resize(image, (new_h, new_w))
        return image

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def blob_for_bbox(bbox, heatmap, downsampling_factor=None):
    if downsampling_factor is not None:
        left, top, w, h = [bbox_coord // downsampling_factor for bbox_coord in bbox]
    else:
        left, top, w, h = [bbox_coord for bbox_coord in bbox]

    right, bottom = left+w, top+h
    ct_int = None
    if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        heatmap = draw_umich_gaussian(heatmap, ct_int, radius)
    return heatmap, ct_int

def pre_process_centernet(image, meta=None, fix_res=True):
    scale = 1.0
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if fix_res:
        inp_height, inp_width = 512, 512
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
    else:
        inp_height = (new_height | 31) + 1
        inp_width = (new_width | 31) + 1
        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    # if self.opt.flip_test:
    #     images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    # meta = {'c': c, 's': s,
    #         'out_height': inp_height // self.opt.down_ratio,
    #         'out_width': inp_width // self.opt.down_ratio}
    return images.squeeze() #, meta

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def load_checkpoint(model, trained_model_weights_filename):
    checkpoint = torch.load(trained_model_weights_filename, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    return model

def load_model(arch, model_weights, device):

    if model_weights is None:
        if arch == 'mobilenet_v3_small':
            model_weights = 'models/mobilenet_v3_pretrained.pth'
            arch = 'mobilenetv3small'
        elif arch == 'res_18':
            model_weights = 'models/res18_pretrained.pth'
        elif arch == 'dla_34':
            model_weights = 'models/dla_34_pretrained.pth'

    heads = {'hm':1} if arch != 'dla_34' else {'hm':1, 'wh':2}

    model = create_base(arch, heads=heads, head_conv=256).to(device)
    model = load_checkpoint(model, model_weights)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model

def _calculate_euclidean_similarity(distances, zero_distance):
    """ Calculates the euclidean distance between two sets of detections, and then converts this into a similarity
    measure with values between 0 and 1 using the following formula: sim = max(0, 1 - dist/zero_distance).
    The default zero_distance of 2.0, corresponds to the default used in MOT15_3D, such that a 0.5 similarity
    threshold corresponds to a 1m distance threshold for TPs.
    """
    sim = np.maximum(0, 1 - distances/zero_distance)
    return sim
