. scripts_for_experiments/shell_variables.sh 

experiment_name='new_synthetic_videos_val_CenterTrack_dets_2_Kalman'
output_dir='experiments/tracking/'${experiment_name}
external_detections_dir='data/detector_results/real_val/CenterTrack'
create_clean_directory $output_dir 

python -m debugpy --listen 5678 --wait-for-client track.py \
    --data_dir '/home/mathis/Documents/datasets/surfrider/videos/gopro_video/true_validation/true_val' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.33 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 5 \
    --external_detections_dir ${external_detections_dir} \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --detector 'external_pickle' \
    --tracker_parameters_dir 'data/tracking_parameters'








