import cv2
import numpy as np
import os
from detection.detect import detect
from tools.misc import load_model
from PIL import Image
from detection.transforms import TransformFrames, TrainTransforms
import torch
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch._C import MobileOptimizerType
from load_dataset import create_input_example
from random import sample
import json



def main(args):

    print('---Loading model...')

    model = torch.jit.load("models/S_MobileNet_SO4.pt")

    print("Quantization ok")

    with open("data/images/dict_name.json") as f:
        dict_name = json.load(f)
    
    

    print('Model loaded.')

    def detector(frame): return detect(frame, threshold=args.detection_threshold,
                                            model=model)

    # with open("data/images/instances_val.json") as f:
    #     data = json.load(f)
    # frame_filenames = [x["file_name"] for x in data["images"]]


    # frame_filenames = [name for name in os.listdir('data/images/images/validation_images')]
    frame_filenames = ["9ca06b40bb401ab71f71f7e4b9cf9013.jpeg"]
    result = {}

    for frame_filename in frame_filenames: 
        print(frame_filename)
        frame = Image.open(os.path.join("data/images/",frame_filename))
        input_shape = frame.size
        frame = np.asarray(frame)

        display_frame = frame

        print('Detections...')
        
        frame = TransformFrames()(frame)[None, :]
        print(frame.shape)        

        detection_for_frame = detector(frame)


        detection_for_frame = detection_for_frame[0][0]



        # result[dict_name[frame_filename]] = detection_for_frame.numpy().tolist()

        if len(detection_for_frame) : 
            # detection_for_frame = detection_for_frame
            result_frame = display_frame 
            for detection in detection_for_frame:
                # print(detection)

                # ratio_y =  input_shape[0]/ (args.output_shape[0] // 4)
                # ratio_x = input_shape[1] / (args.output_shape[1] // 4)

                # detection_with_ratio = (detection_for_frame[0]* ratio_x, detection_for_frame[1]* ratio_y)
                detection_with_ratio = (detection[0]* 4, detection[1]* 4)
                # display_frame = np.asarray(display_frame.resize(args.output_shape))

                result_frame = cv2.circle(result_frame, (int(detection_with_ratio[0]), int(detection_with_ratio[1])), 5, (0, 0, 255), -1)
            cv2.imshow("image", result_frame)
            cv2.waitKey(0)

        else: print("Pas de déchet trouvé")

    # with open("heatmap2.json", 'w') as fp:
    #     json.dump(result, fp)  


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--detection_threshold', type=float, default=0.33)
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--arch', type=str, default='dla_34')

    args = parser.parse_args()


    main(args)
