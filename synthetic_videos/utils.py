import cv2
import numpy as np
import flow_tools
from scipy.stats import multivariate_normal
from scipy.signal import savgol_filter


class OverlayTrash():
    def __init__(self, trash_image, init_loc, final_loc, 
              size, init_frame, final_frame):
        
        self.trash_image = trash_image
        self.init_loc, self.final_loc = init_loc, final_loc
        self.init_frame, self.final_frame = init_frame, final_frame
        self.span_x = (final_loc[0] - init_loc[0])
        self.span_y = (final_loc[1] - init_loc[1])
        self.length = final_frame - init_frame
        self.reshape_size = (size, int(size*trash_image.shape[0]/trash_image.shape[1]))
        self.size = size
        
    def get_position(self, frame_idx):
        alpha = (frame_idx - self.init_frame) / self.length
        x = int(self.init_loc[0] + self.span_x * alpha)
        y = int(self.init_loc[1] + self.span_y * alpha)
        return (x,y)

def blockshaped(arr, nrows, ncols):

    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
        .swapaxes(1,2)
        .reshape(-1, nrows, ncols))

def update_points_dense(points, flow_dense):
    for point in points:
        point[1] = point[1] + int(flow_dense[point[1],point[0],1])
        point[0] = point[0] + int(flow_dense[point[1],point[0],0])
    return points 

def blob_for_bbox(img, bbox):
    [top_left_x, top_left_y, width, height] = bbox

    bbox_coords = [[top_left_x, top_left_y],
                   [top_left_x+width, top_left_y],
                   [top_left_x+width, top_left_y+height],
                   [top_left_x, top_left_y+height]]
    bbox_center = np.mean(bbox_coords,axis=0)
    bbox_cov = np.diag([width, height])

    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x2d, y2d = np.meshgrid(x, y)
    pos = np.dstack((x2d, y2d))
    rv = multivariate_normal(mean=bbox_center,cov=bbox_cov)
    blob = rv.pdf(pos)
    # fig, ax = plt.subplots(1,1, figsize=(10,10))
    # ax.imshow(temp, cmap='gray')
    # ax.imshow(img, alpha=0.1)
    # plt.show()
    # temp /= temp.max()
    # gray = (255 * temp).astype(np.uint8)
    # return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return blob

def overlay_transparent(background_img, img_to_overlay_t, alpha, x, y, overlay_size=None):


    bg_img = background_img.copy()
    background_height, background_width, _ = bg_img.shape
    
    if overlay_size is not None:
        overlay_size = (overlay_size[1],overlay_size[0])
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB 
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))
    
    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a,5)

    # Border conditions
    h, w, _ = overlay_color.shape

    if x < 0:
        w = w + x
        mask = mask[:, -x:]
        overlay_color = overlay_color[:, -x:]
        x = 0
    
    if y < 0:
        h = h + y
        mask = mask[-y:, :]
        overlay_color = overlay_color[-y:, :]
        y = 0
    
    if x + w > background_width:
        w = background_width - x
        mask = mask[:, :w]
        overlay_color = overlay_color[:, :w]

    if y + h > background_height:
        h = background_height - y
        mask = mask[:h, :]
        overlay_color = overlay_color[:h, :]

    if h > 0 and w > 0: 

        roi = bg_img[y:y+h, x:x+w]

        # Black-out the area behind the overlay in our original ROI
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask = cv2.bitwise_not(mask))

        # Mask out the overlay from the logo image.
        img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask = mask)

        # Update the original image with our new ROI
        img_with_overlay = bg_img.copy()
        img_with_overlay[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

        result = cv2.addWeighted(bg_img, 1-alpha, img_with_overlay, alpha, 0)

        bbox = [x,y,w,h]

        return result, bbox
    else: 
        return None, None

def overlay_trash(img, trash, alpha, pts, shape):
    tracked_pos_y = int(np.mean(pts[:,1]))
    tracked_pos_x = int(np.mean(pts[:,0]))
    center_x = tracked_pos_x - int(shape[0]/2)
    center_y = tracked_pos_y - int(shape[1]/2)
    return [center_x, center_y], overlay_transparent(img, trash, alpha, center_x, center_y, shape)

def displacement_norm(pts_sequence, use_dense=False, img=None, img2=None):
    if not use_dense:
        return np.abs(np.mean(pts_sequence[-1])-np.mean(pts_sequence[-2]))
    else:
        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(pts_sequence[-1])
        img = img[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
        img2 = img2[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
        flow_dense_roi = flow_tools.flow_opencv_dense(img,img2)
        mag, ang = cv2.cartToPolar(flow_dense_roi[...,0], flow_dense_roi[...,1])
        return mag

def spread(points):
    dist_to_mean = points-np.mean(points,axis=0)
    if len(points) > 1: spread = np.linalg.norm(dist_to_mean,ord=2,axis=1).mean() / len(points)
    else: spread = np.linalg.norm(dist_to_mean,ord=2)
    return spread 

def clean_displacement_norm_sequence(displacement_norm_sequence):

    # odd_half_window = int(len(displacement_norm_sequence)/2)
    # odd_half_window = odd_half_window + 1 if (odd_half_window % 2 == 0) else odd_half_window
    displacement_norm_sequence = np.convolve(displacement_norm_sequence, np.ones(9)/9, mode='same')
    displacement_norm_sequence =  displacement_norm_sequence/displacement_norm_sequence[0]
    displacement_norm_sequence[displacement_norm_sequence > 2] = 2
    displacement_norm_sequence[displacement_norm_sequence < 0.5] = 0.5
    return np.sqrt(displacement_norm_sequence)

def rescaling(ratio, original_shape):
    return (int(ratio * original_shape[0]), int(ratio * original_shape[1])) 

def moving_average(displacement_norm_sequence, window_size):
    half_odd_window = window_size // 2 + 1
    new_displacement_norm_sequence = [] 
    for i in range(len(displacement_norm_sequence)):
        left = max(0, i-half_odd_window)
        right = min(len(displacement_norm_sequence),i+half_odd_window+1)
        new_displacement_norm_sequence.append(np.mean(displacement_norm_sequence[left:right]))
    return new_displacement_norm_sequence