. scripts/shell_variables.sh 

algorithm=EKF_1
details='viz'
experiment_name=${algorithm}_${details}
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir


python src/track.py \
    --data_dir data/validation_videos/all/short_segments_12fps/videos \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --detection_threshold 0.38 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --algorithm ${algorithm} \
    --noise_covariances_path data/tracking_parameters \
    --model_weights models/centernet_pretrained.pth \
    --output_shape 960,544 \
    --skip_frames 0 \
    --display 2



