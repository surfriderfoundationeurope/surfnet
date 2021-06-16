import numpy as np 
from common.opencv_tools import SimpleVideoReader
import cv2
import os 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_dir = 'data/validation_videos/T1/long_segments'

# mot_results2  = np.loadtxt('gt2.txt', delimiter=',')

# num_frames_1 = 6474

# mot_results2[:,0]+=num_frames_1
# mot_results2[:,1]+=mot_results1[:,1].max()

# mot_results = np.concatenate([mot_results1,mot_results2])

seqmaps_dir = os.path.join(output_dir,'seqmaps')
os.mkdir(seqmaps_dir)
seqmaps = open(os.path.join(seqmaps_dir,'surfrider-test.txt'),'w')
seqmaps.write('name\n')
sequences_dir = os.path.join(output_dir,'surfrider-test')
os.mkdir(sequences_dir)

mot_results1  = np.loadtxt('data/validation_videos/T1/CVAT/gt1.txt', delimiter=',')
# segments_ends = [853,1303,1984,2818,3509,4008,4685,5355,np.inf]
segments_ends = [np.inf]
video = SimpleVideoReader('data/validation_videos/T1/CVAT/part1.mp4', skip_frames=0)
largest_index = 0
for segment_nb, segment_end in enumerate(segments_ends):
    sequence_len = 0
    sequence_name = 'part_1_segment_{}'.format(segment_nb)
    sequence_dir = os.path.join(sequences_dir,sequence_name)
    os.mkdir(sequence_dir)
    seqmap_file = open(os.path.join(sequence_dir,'seqinfo.ini'),'w')

    sequence_gt_dir = os.path.join(sequence_dir,'gt')
    os.mkdir(sequence_gt_dir)

    first_frame_added = None
    writer = cv2.VideoWriter(filename=os.path.join(output_dir,sequence_name+'.mp4'), apiPreference=cv2.CAP_FFMPEG, fourcc=fourcc, fps=video.fps/2, frameSize=video.shape, params=None)
    while True: 
        ret, frame, frame_read_nb = video.read()
        if ret and (frame_read_nb + 1 <= segment_end):
            if frame_read_nb % 2 == 0: 
                if first_frame_added is None: first_frame_added = frame_read_nb
                writer.write(frame)
                sequence_len+=1
        else: 
            break
        
    writer.release()
    with open(os.path.join(sequence_gt_dir,'gt.txt'),'w') as out_file:
        mot_results_for_segment = [mot_result for mot_result in mot_results1 if (mot_result[0] <= segment_end and mot_result[0] >= first_frame_added+1)]
        for mot_result in mot_results_for_segment:
            frame_id = int(mot_result[0])
            track_id = int(mot_result[1])
            left, top, width, height = mot_result[2:6]
            center_x = left+width/2
            center_y = top+height/2
            if (frame_id - 1) % 2 == 0: 
                frame_id_to_write = int((frame_id - 1) // 2 - first_frame_added // 2 + 1)
                track_id_to_write = int(track_id - largest_index)
                out_file.write('{},{},{},{},{},{},{},{}\n'.format(frame_id_to_write,
                                                                track_id_to_write,
                                                                center_x,
                                                                center_y,
                                                                -1,
                                                                -1,
                                                                1,
                                                                -1))
        largest_index = max(mot_result[1] for mot_result in mot_results_for_segment)


    seqmap_file.write('[Sequence]\nname={}\nimDir=img1\nframeRate={}\nseqLength={}\nimWidth=1920\nimHeight=1080\nimExt=.png'.format(sequence_name,video.fps/2,sequence_len))
    seqmap_file.close()
    seqmaps.write(sequence_name)


mot_results2  = np.loadtxt('data/validation_videos/T1/CVAT/gt2.txt', delimiter=',')
# segments_ends = [844,2021,2692,3544,3999,4744,5171,6127,6889,np.inf]
segments_ends = [np.inf]
video = SimpleVideoReader('data/validation_videos/T1/CVAT/part2.mp4', skip_frames=0)
largest_index = 0
for segment_nb, segment_end in enumerate(segments_ends):
    sequence_len = 0
    sequence_name = 'part_2_segment_{}'.format(segment_nb)
    sequence_dir = os.path.join(sequences_dir,sequence_name)
    os.mkdir(sequence_dir)
    seqmap_file = open(os.path.join(sequence_dir,'seqinfo.ini'),'w')

    sequence_gt_dir = os.path.join(sequence_dir,'gt')
    os.mkdir(sequence_gt_dir)

    first_frame_added = None
    writer = cv2.VideoWriter(filename=os.path.join(output_dir,sequence_name+'.mp4'), apiPreference=cv2.CAP_FFMPEG, fourcc=fourcc, fps=video.fps/2, frameSize=video.shape, params=None)
    while True: 
        ret, frame, frame_read_nb = video.read()
        if ret and (frame_read_nb + 1 <= segment_end):
            if frame_read_nb % 2 == 0: 
                if first_frame_added is None: first_frame_added = frame_read_nb
                writer.write(frame)
                sequence_len+=1
        else: 
            break
        
    writer.release()
    with open(os.path.join(sequence_gt_dir,'gt.txt'),'w') as out_file:
        mot_results_for_segment = [mot_result for mot_result in mot_results2 if (mot_result[0] <= segment_end and mot_result[0] >= first_frame_added+1)]
        for mot_result in mot_results_for_segment:
            frame_id = int(mot_result[0])
            track_id = int(mot_result[1])
            left, top, width, height = mot_result[2:6]
            center_x = left+width/2
            center_y = top+height/2
            if (frame_id - 1) % 2 == 0: 
                frame_id_to_write = int((frame_id - 1) // 2 - first_frame_added // 2 + 1)
                track_id_to_write = int(track_id - largest_index)
                out_file.write('{},{},{},{},{},{},{},{}\n'.format(frame_id_to_write,
                                                                track_id_to_write,
                                                                center_x,
                                                                center_y,
                                                                -1,
                                                                -1,
                                                                1,
                                                                -1))
        largest_index = max(mot_result[1] for mot_result in mot_results_for_segment)


    seqmap_file.write('[Sequence]\nname={}\nimDir=img1\nframeRate={}\nseqLength={}\nimWidth=1920\nimHeight=1080\nimExt=.png'.format(sequence_name,video.fps/2,sequence_len))
    seqmap_file.close()
    seqmaps.write(sequence_name+'\n')

seqmaps.close()

