import shutil
import os 
results_dir = 'surfrider_short_segments_6fps'
target_dir = '/home/infres/chagneux/repos/TrackEval/data/trackers/surfrider_short_segments_6fps/surfrider-test/fairmot_cleaned/data'
sequence_names = [sequence_name for sequence_name in next(os.walk(results_dir))[1]]

for sequence_name in sequence_names:
    shutil.copy(os.path.join(results_dir,sequence_name,'results_clean_1.txt'), os.path.join(target_dir,sequence_name+'.txt'))
