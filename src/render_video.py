import cv2 
import os
from tqdm import tqdm
images_dir = 'plots'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

filenames = sorted([os.path.join(images_dir, filename) for filename in os.listdir(images_dir)],key=lambda x:int(x.split('/')[1].split('.')[0]))
writer = cv2.VideoWriter(filename=os.path.join(images_dir, 'render.mp4'), 
                            apiPreference=cv2.CAP_FFMPEG, 
                            fourcc=fourcc, 
                            fps=11.98, 
                            frameSize= cv2.imread(filenames[0]).shape[:-1][::-1], 
                            params=None)

for filename in tqdm(filenames):
    writer.write(cv2.imread(filename))

writer.release()