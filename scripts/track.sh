. scripts/shell_variables.sh 

experiment_name='tracking_best_T3'
output_dir='experiments/tracking/'${experiment_name}
create_clean_directory $output_dir 

python src/track.py \
    --data_dir 'data/validation_videos/T3' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.4 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 5 \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --detector 'internal_base' \
    --tracker_parameters_dir 'data/tracking_parameters' \
    --base_weights 'experiments/base/dla_34_downsample_4_alpha_2_beta_4_lr_1.25e-4_batch_size_32_single_class_rectangular_shape/model_139.pth' \
    --output_w 960 \
    --output_h 544 \
    --skip_frames 1






