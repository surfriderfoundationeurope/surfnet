import os 
import shutil 

split = 'train'
data_dir = 'src/data/surfrider'
old_images_dir = os.path.join(data_dir,'images')
new_images_dir = os.path.join(old_images_dir, split)
old_labels_dir = os.path.join(data_dir,'labels_with_ids')
new_labels_dir = os.path.join(old_labels_dir, split)
# os.mkdir(new_images_dir)
# os.mkdir(new_labels_dir)

filenames_path = 'src/data/surfrider.{}'.format(split)

with open(filenames_path,'r') as f:
    filenames = [filename.split('/')[1].strip('\n') for filename in f.readlines()]

for filename in filenames: 
    # shutil.move(os.path.join(old_images_dir,filename),os.path.join(new_images_dir,filename))
    shutil.move(os.path.join(old_labels_dir,filename.replace('.jpg','.txt')),os.path.join(new_labels_dir,filename.replace('.jpg','.txt')))

