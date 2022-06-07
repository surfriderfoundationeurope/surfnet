import cv2


def flow_opencv_dense(img, img2):

    prvs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow


def compute_flow(frame0, frame1, downsampling_factor):

    if downsampling_factor > 1:
        h, w = frame0.shape[:-1]
        new_h = h // downsampling_factor
        new_w = w // downsampling_factor

        frame0 = cv2.resize(frame0, (new_w, new_h))
        frame1 = cv2.resize(frame1, (new_w, new_h))

    return flow_opencv_dense(frame0, frame1)
