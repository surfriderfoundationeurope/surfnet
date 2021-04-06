import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from synthetic_videos import taco_tools, cv2_io, flow_tools, utils
import json

verbose = False


def init_tracker(first_frame, dense_flow_first_frame, object_nb, tractable_band='any'):

    v, u = dense_flow_first_frame[...,1], dense_flow_first_frame[...,0]
    norm = np.sqrt(u ** 2 + v ** 2)

    pgcd = np.gcd(norm.shape[0], norm.shape[1])
    block_size = (int(norm.shape[0]/pgcd), int(norm.shape[1]/pgcd))

    norm_blocks = np.mean(utils.blockshaped(norm, pgcd, pgcd),axis=(1,2)).reshape(block_size)

    if tractable_band == 'center_left':
        norm_tractacle_region = norm[int(3*norm.shape[0]/6):int(4*norm.shape[0]/6),:int(norm.shape[1]/3)]
        norm_blocks_tractacle_regions = norm_blocks[int(3*block_size[0]/6):int(4*block_size[0]/6),:int(block_size[1]/3)]
    elif tractable_band == 'lower_half_left': 
        norm_tractacle_region = norm[int(norm.shape[0]/2):,:int(norm.shape[1]/3)]
        norm_blocks_tractacle_regions = norm_blocks[int(block_size[0]/2):,:int(block_size[1]/3)]
    else:
        norm_tractacle_region = norm[int(norm.shape[0]/2):,:]
        norm_blocks_tractacle_regions = norm_blocks[int(block_size[0]/2):,:]

    mean_norm = np.mean(norm_tractacle_region)
    
    dist_to_median = np.abs(norm_blocks_tractacle_regions-(object_nb+1)*mean_norm)
    median_block = np.array(np.unravel_index(np.argmin(dist_to_median, axis=None), dist_to_median.shape)) + np.array([int(block_size[0]/2),0])

    roi = pgcd*np.array([median_block[1],median_block[1]+1,median_block[0],median_block[0]+1])
    
    mask = np.zeros_like(first_frame)
    mask[roi[2]:roi[3], roi[0]:roi[1]] = 255

    p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), mask = mask[:,:,0], **flow_tools.feature_params)
    
    if verbose:

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 4))
        img = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        img[roi[2]:roi[3],roi[0]:roi[1]] = np.inf
        ax0.imshow(img)
        ax0.set_axis_off()

        ax1.imshow(norm_blocks)
        ax1.plot(median_block[1], median_block[0],'ro')
        ax1.set_axis_off()
        ax2.imshow(norm)
        ax2.set_axis_off()
        plt.show()
    return p0

def condition_tracking(points, init_nb_pts):
    if len(points) != init_nb_pts: 
        print('Lost one point.')
        return False
    else: 
        spread_pts = utils.spread(points)
        spread_too_high = (spread_pts > 2)
        if spread_too_high: print('Points spreaded too much.')
        return (not spread_too_high)
        
def compute_flow(frame_reader, tractable_band, object_nb):

    ret, first_frame = frame_reader.read_frame()
    if not ret: return [], []
    ret, second_frame = frame_reader.read_frame()
    if not ret: return [], []
    dense_flow_first_frame = flow_tools.flow_opencv_dense(first_frame, second_frame)
    pts_sequence = list()
    p0 = init_tracker(first_frame, dense_flow_first_frame, object_nb, tractable_band)
    if p0 is not None: 
        pts_sequence.append(p0.squeeze())

        init_nb_pts = len(pts_sequence[-1])
        print('Init nb points: {}'.format(init_nb_pts))
        if pts_sequence[-1].ndim < 2: 
            return [], []

        init_spread = utils.spread(pts_sequence[-1]) 
        print('Init spread: {}'.format(init_spread))
        if init_spread > 2:
            return [], []
    else: 
        print('Init nb points: {}'.format(0))
        return [], []

    new_pts = flow_tools.flow_opencv_sparse(first_frame, second_frame, p0)
    pts_sequence.append(new_pts)

    displacement_norm_sequence = list()
    displacement_norm_sequence.append(utils.displacement_norm(pts_sequence))

    old_frame = second_frame.copy()
    p0 = new_pts.reshape(-1,1,2)
    while(condition_tracking(pts_sequence[-1], init_nb_pts)):
        ret, frame = frame_reader.read_frame()
        if not ret: return [], []
        new_pts = flow_tools.flow_opencv_sparse(old_frame, frame, p0)
        pts_sequence.append(new_pts)
        displacement_norm_sequence.append(utils.displacement_norm(pts_sequence))
        old_frame = frame.copy()
        p0 = new_pts.reshape(-1,1,2)
    return pts_sequence, displacement_norm_sequence

def compute_flows(frame_reader, tractable_band, nb_objects): 
    pts_sequences = []
    displacement_norm_sequences = []
    for object_nb in range(nb_objects):
        pts_sequence, displacement_norm_sequence = compute_flow(frame_reader, tractable_band, object_nb=object_nb)
        pts_sequences.append(pts_sequence)
        displacement_norm_sequences.append(displacement_norm_sequence) 
        frame_reader.reset_init_frame()
    return pts_sequences, displacement_norm_sequences
        
def add_trash_objects(frame_reader, frame_writer, json_file, tractable_band, nb_objects, output_original_shape=False):
    
    pts_sequences, displacement_norm_sequences = compute_flows(frame_reader, tractable_band, nb_objects=nb_objects)

    usable_pts_sequences = []
    usable_displacement_norm_sequences = []
    trash_images = []
    alphas = []
    init_shapes = []
    annotations = dict()


    for pts_sequence, displacement_norm_sequence in zip(pts_sequences,displacement_norm_sequences):
        if len(pts_sequence) > 10:
            usable_pts_sequences.append(pts_sequence)
            usable_displacement_norm_sequences.append(utils.clean_displacement_norm_sequence(displacement_norm_sequence))
            trash_images.append(taco_tools.get_random_trash(label="bottle", anns=anns, imgs=imgs, dict_label_to_ann_ids=dict_label_to_ann_ids))
            alphas.append(random.uniform(0.8,1))
    if not len(usable_pts_sequences): return 100

    shortest_sequence_length = min([len(x) for x in usable_pts_sequences])

    for frame_nb in range(shortest_sequence_length):
        annotations[frame_nb] = dict()
    
    
    if output_original_shape:
        frame_reader.set_original_shape_mode(True)
        init_area = (frame_reader.original_width * frame_reader.original_height) / 500 
        for i, pts_sequence in enumerate(usable_pts_sequences):
            usable_pts_sequences[i] = [frame_reader.init_rescale_factor * pts for pts in pts_sequence]
    else: 
        init_area = (frame_reader.new_shape[0] * frame_reader.new_shape[1]) / 500

    for object_nb in range(len(usable_pts_sequences)):
        trash_img = trash_images[object_nb]
        trash_area = trash_img.shape[0] * trash_img.shape[1]
        randomizer_init_area = random.uniform(0.8,1)
        init_coeff_reshape = np.sqrt(randomizer_init_area*init_area/trash_area)
        # print(trash_area)
        init_shapes.append(utils.rescaling(init_coeff_reshape, trash_img.shape))


    for frame_nb in range(shortest_sequence_length):

        _ , frame = frame_reader.read_frame()

        for object_nb in range(len(usable_pts_sequences)):

            pts_sequence = usable_pts_sequences[object_nb]
            displacement_norm_sequence = usable_displacement_norm_sequences[object_nb]
            # print(displacement_norm_sequence)
            trash_img = trash_images[object_nb]
            alpha = alphas[object_nb]
            init_shape = init_shapes[object_nb]
            # if np.isnan(min(frame_nb,len(displacement_norm_sequence)-1)): print('BOOH')
            if np.any(np.isnan(displacement_norm_sequence)): 
                break
            shape = utils.rescaling(displacement_norm_sequence[min(frame_nb,len(displacement_norm_sequence)-1)], init_shape)
            center, (frame, bbox) = utils.overlay_trash(frame, trash_img, alpha, pts_sequence[frame_nb], shape)
            if frame is not None:
                annotations[frame_nb][object_nb] = {'bbox': bbox, 'center': center}
            else: 
                return frame_nb
        frame_writer.write(frame)

    if frame_reader.original_shape_mode: frame_reader.set_original_shape_mode(False)
    json.dump(annotations, json_file)
    return shortest_sequence_length



def main(args):

    output_original_shape = args.original_res
    vid_dir = args.vid_dir
    tractable_band = args.tractable_band
    vid_names = [name for name in os.listdir(vid_dir) if '.MP4' in name]
    video_filenames = [vid_dir + vid_name for vid_name in vid_names if '.MP4' in vid_name]
    rescale_factor = args.rescale
    nb_extracts_per_vid = args.nb_extracts_per_vid
    output_dir = args.output_dir
    read_every = args.read_every

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for video_nb, video_filename in enumerate(video_filenames):

        frame_reader = cv2_io.FrameReader(video_filename, read_every=read_every, rescale_factor=rescale_factor, init_time_min=0, init_time_s=10)
        fps = frame_reader.fps
        print(fps)
        shape = (int(frame_reader.original_width), int(frame_reader.original_height))
        if not output_original_shape: shape = frame_reader.new_shape 
        total_frames_read = frame_reader.init_frame
        total_num_frames = frame_reader.total_num_frames
        portion_nb_in_video=0
        while total_frames_read < (total_num_frames - 100): 

            frame_reader.set_init_frame(total_frames_read)
            output_name = output_dir + vid_names[video_nb].strip('.MP4') + '_{}.MP4'.format(portion_nb_in_video)
            json_file = open(output_name.replace('.MP4','.json'),'w')
            nb_objects = 2
            frame_writer = cv2.VideoWriter(filename=output_name, apiPreference=cv2.CAP_FFMPEG, fourcc=fourcc, fps=fps, frameSize=shape)

            nb_frames_read = add_trash_objects(frame_reader, frame_writer, json_file, tractable_band=tractable_band, nb_objects=nb_objects, output_original_shape=True)

            total_frames_read += nb_frames_read + int(total_num_frames / nb_extracts_per_vid)
            portion_nb_in_video+=1
            json_file.close()
            frame_writer.release()

        # if args.nb_frames_without_object: 

        #     frame_reader = cv2_io.FrameReader(video_filename, read_every=read_every, rescale_factor=rescale_factor, init_time_min=0, init_time_s=10)
        #     if output_original_shape:
        #             frame_reader.set_original_shape_mode(True)
        #     output_name = output_dir + vid_names[video_nb].replace('.MP4','_no_object.MP4')
        #     frame_writer = cv2.VideoWriter(filename=output_name, apiPreference=cv2.CAP_FFMPEG, fourcc=fourcc, fps=fps, frameSize=shape)
        #     json_file = open(output_name.replace('.MP4','.json'),'w')
        #     annotations = dict()
        #     for frame_nb in range(args.nb_frames_without_object):
        #         _ , frame =  frame_reader.read_frame()
        #         # cv2.imshow('frame', frame)
        #         # cv2.waitKey(0)
        #         frame_writer.write(frame)
        #         annotations[frame_nb] = dict()
        #     json.dump(annotations, json_file)
        #     json_file.close()
        #     frame_writer.release()
    

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Surfnet training')
    parser.add_argument('--vid-dir', dest='vid_dir',type=str)
    parser.add_argument('--tractable-band',default='any',type=str)
    parser.add_argument('--output-dir',dest='output_dir',type=str)
    parser.add_argument('--read-every',dest='read_every',default=2, type=int)
    parser.add_argument('--original-res',dest='original_res',action='store_true')
    parser.add_argument('--rescale', type=int, default=3)
    parser.add_argument('--nb-extracts-per-vid',dest='nb_extracts_per_vid', type=int, default=20)
    parser.add_argument('--synthetic-objects',type=str)
    parser.add_argument('--nb-frames-without-object', type=int, dest='nb_frames_without_object', default=50)
    parser.add_argument('--max-nb-objects',dest='max_nb_objects',type=int,default=3)

    args = parser.parse_args()
    taco_path = args.synthetic_objects
    taco_tools.taco_path = taco_path
    anns, imgs, dict_label_to_ann_ids = taco_tools.load_TACO()

    main(args)


    







