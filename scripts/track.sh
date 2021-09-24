. scripts/shell_variables.sh 

algorithm=EKF
details=''
experiment_name=${algorithm}_${details}
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir


python src/track.py \
    --data_dir data/validation_videos \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --detection_threshold 0.38 \
    --downsampling_factor 4 \
    --noise_covariances_path data/tracking_parameters \
    --model_weights /home/mathis/repos/surfnet/models/centernet_pretrained.pth \
    --output_shape 960,544 \
    --skip_frames 1 \
    --display 1



