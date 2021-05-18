. scripts_for_experiments/shell_variables.sh 

experiment_name='real_vid_centertrack'
output_dir='experiments/tracking/'${experiment_name}
external_detections_dir='data/detector_results/real_val/CenterTrack'
create_clean_directory $output_dir 

python track.py \
    --data_dir '/home/infres/chagneux/datasets/surfrider_data/video_dataset/true_validation/T1' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.33 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 5 \
    --external_detections_dir ${external_detections_dir} \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --detector 'external_CenterTrackpickle' \
    --tracker_parameters_dir 'data/tracking_parameters'








