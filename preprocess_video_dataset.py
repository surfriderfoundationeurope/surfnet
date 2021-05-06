from tqdm import tqdm
import cv2
import os
import json


class VideoOpenCV(object):

    def __init__(self, video_name):
        self.cap = cv2.VideoCapture(video_name)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            print('Unreadable frame!')

        return frame


def main(args):
    video_folder = args.input_dir
    video_names = [video_name for video_name in sorted(
        os.listdir(video_folder)) if '.MP4' in video_name]

    for video_name in tqdm(video_names):
        folder_for_video = os.path.join(args.output_dir, 'data', video_name.split('.')[0])
        os.mkdir(folder_for_video)
        video = VideoOpenCV(video_folder + video_name)
        num_frames = video.num_frames

        for frame_nb in range(1, num_frames+1):
            frame = video.read()
            cv2.imwrite(os.path.join(folder_for_video,
                        '{:03d}.png'.format(frame_nb)), frame)

    with open(os.path.join(args.input_dir,'annotations_val.json'), 'r') as f:
        annotations = json.load(f)

    for video in annotations['videos']:
        video['file_name'] = video['file_name'].split('.')[0]

    for image in annotations['images']:
        image['file_name'] = image['file_name'].split(
            '.')[0]+'/{:03d}.png'.format(image['frame_id'])

    annotations_path = os.path.join(args.output_dir,'annotations')
    # os.mkdir(annotations_path)

    with open(os.path.join(annotations_path, 'annotations_val.json'), 'w') as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    main(args)
