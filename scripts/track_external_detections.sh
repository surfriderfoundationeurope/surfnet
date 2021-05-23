. scripts/shell_variables.sh 

experiment_name='T1_augmentations'
output_dir='experiments/tracking/'${experiment_name}

create_clean_directory $output_dir 

python src/track.py \
    --data_dir ${VALIDATION_VIDEOS}/T1 \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.33 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 5 \
    --external_detections_dir ${EXTERNAL_DETECTIONS}/real_val/mine/threshold_04_augmentations \
    --algorithm Kalman \
    --read_from folder \
    --detector external_simplepickle \
    --tracker_parameters_dir data/tracking_parameters \
    --output_w 960 \
    --output_h 544 \
    --skip_frames 1
