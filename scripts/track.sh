. scripts/shell_variables.sh 

experiment_name='T1_long_segments_trained_new_data_2nd_train_threshold_028_clean_1'
output_dir='experiments/tracking/'${experiment_name}
create_clean_directory $output_dir

python src/track.py \
    --data_dir 'data/validation_videos/T1/CVAT' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.28 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 1 \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --annotation_file 'data/synthetic_videos/annotations/annotations_val.json' \
    --detector 'internal_base' \
    --tracker_parameters_dir 'data/tracking_parameters' \
    --base_weights 'models/centernet_newdata_pretrained_different_resize_epoch_140.pth' \
    --output_w 960 \
    --output_h 544 \
    --skip_frames 1



