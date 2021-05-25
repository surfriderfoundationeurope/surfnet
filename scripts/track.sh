. scripts/shell_variables.sh 

experiment_name='epoch_290'
output_dir='experiments/tracking/'${experiment_name}
create_clean_directory $output_dir 

python src/track.py \
    --data_dir 'data/validation_videos/T1' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.5 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 5 \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --detector 'internal_base' \
    --tracker_parameters_dir 'data/tracking_parameters' \
    --base_weights 'experiments/base/290_epochs_all_augmentations/model_289.pth' \
    --output_w 960 \
    --output_h 544 \
    --skip_frames 1






