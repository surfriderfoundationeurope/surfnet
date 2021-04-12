import os 
video_folder = './data/generated_videos/'
video_names = [video_name for video_name in sorted(os.listdir(video_folder)) if '.MP4' in video_name]
import json 

# print(video_names)
vid_number_to_vid_name = dict()
for video_nb, video_name in enumerate(video_names):
    vid_number_to_vid_name['video_{:03d}'.format(video_nb)] = video_name

with open('vid_nb_to_vid_names.json','w') as f:
    json.dump(vid_number_to_vid_name,f)
