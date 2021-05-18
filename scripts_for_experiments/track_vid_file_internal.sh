. scripts_for_experiments/shell_variables.sh 

experiment_name='vid_file_internal_base_halffps_960_544'
output_dir='experiments/tracking/'${experiment_name}
create_clean_directory $output_dir 

python track.py \
    --data_dir '/home/infres/chagneux/datasets/surfrider_data/video_dataset/true_validation/T1' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.3 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 5 \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --detector 'internal_base' \
    --tracker_parameters_dir 'data/tracking_parameters' \
    --base_weights 'experiments/base/dla_34_downsample_4_alpha_2_beta_4_lr_1.25e-4_batch_size_32_single_class_rectangular_shape/model_70.pth' \
    --output_w 960 \
    --output_h 544 \
    --skip_frames 1






