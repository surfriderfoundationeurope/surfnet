from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import pickle

logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')

    if len(opt.input_video):
        dataloader = datasets.LoadVideo(opt.input_video, opt.img_size, skip_frames=0)
    else: 
        dataloader = datasets.LoadImages
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    _, _, _, detections_to_save = eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    with open('saved_frames.pickle','wb') as f:
        pickle.dump(dataloader.frames_to_save, f)

    # with open('saved_heatmaps.pickle','wb') as f:
    #     pickle.dump(heatmaps_to_save, f)

    with open('saved_detections.pickle','wb') as f:
        pickle.dump(detections_to_save, f)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    demo(opt)
