. scripts/shell_variables.sh

experiment_name='prod_'$1
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir

python src/track.py \
    --data_dir 'data/prod_videos/'$1 \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --detection_threshold 0.3 \
    --downsampling_factor 4 \
    --noise_covariances_path data/tracking_parameters \
    --output_shape 960,544 \
    --skip_frames 3 \
    --arch mobilenet_v3_small \
    --device cpu \
    --detection_batch_size 1 \
    --display 0
