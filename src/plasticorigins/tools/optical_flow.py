import cv2
import numpy as np

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 2,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 20,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def viz_dense_flow(frame_reader, nb_frames_to_process):

    ret, frame1 = frame_reader.read_frame()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 0

    while(frame_reader.nb_frames_read < nb_frames_to_process):  
        ret, frame2 = frame_reader.read_frame()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame',np.concatenate([frame2, bgr], axis=0))
        k = cv2.waitKey(0) & 0xff
        prvs = next
    cv2.destroyAllWindows()  

def flow_opencv_dense(img, img2):

    prvs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def flow_opencv_sparse(img, img2, p0):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(img_gray, img2_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]

    return good_new

def dense_flow_norm(dense_flow):
    v, u = dense_flow[...,1], dense_flow[...,0]
    return np.sqrt(u ** 2 + v ** 2)

def compute_flow(frame0, frame1, downsampling_factor):

    if downsampling_factor > 1:
        h, w = frame0.shape[:-1]
        new_h = h // downsampling_factor
        new_w = w // downsampling_factor

        frame0 = cv2.resize(frame0, (new_w, new_h))
        frame1 = cv2.resize(frame1, (new_w, new_h))

    return flow_opencv_dense(frame0, frame1)