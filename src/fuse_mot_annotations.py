import numpy as np 
from common.opencv_tools import SimpleVideoReader
import cv2
mot_results1  = np.loadtxt('gt1.txt', delimiter=',')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# mot_results2  = np.loadtxt('gt2.txt', delimiter=',')

# num_frames_1 = 6474

# mot_results2[:,0]+=num_frames_1
# mot_results2[:,1]+=mot_results1[:,1].max()

# mot_results = np.concatenate([mot_results1,mot_results2])
segments_ends = [853,1303,1984,2818,3509,4008,4685,5355,np.inf]
video = SimpleVideoReader('/media/mathis/EXTERNAL SSD/part1.mp4', skip_frames=0)
largest_index = 0
for segment_nb, segment_end in enumerate(segments_ends):
    writer = cv2.VideoWriter(filename='gt_T1_part_1_segment_{}.mp4'.format(segment_nb), apiPreference=cv2.CAP_FFMPEG, fourcc=fourcc, fps=video.fps/2, frameSize=video.shape, params=None)
    while True: 
        ret, frame, frame_read_nb = video.read()
        if ret and (frame_read_nb + 1 < segment_end):
            if frame_read_nb % 2 == 0: 
                writer.write(frame)
        else: 
            break
    writer.release()
    with open('gt_T1_part_1_segment_{}.txt'.format(segment_nb),'w') as out_file:
        mot_results_for_segment = mot_results1[mot_results1[0] <= segment_end]
        largest_index = mot_results1[:,1].max()
        for mot_result in mot_results_for_segment:
            frame_id = int(mot_result[0])
            track_id = int(mot_result[1])
            left, top, width, height = mot_result[2:6]
            center_x = left+width/2
            center_y = top+height/2
            if (frame_id - 1) % 2 == 0: 
                out_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(int((frame_id - 1 - frame_read_nb) // 2 + 1),
                                                                        int(track_id - largest_index),
                                                                        center_x,
                                                                        center_y,
                                                                        -1,
                                                                        -1,
                                                                        1,
                                                                        -1,
                                                                        -1,
                                                                        -1))




# with open('gt.txt','w') as out_file:


