. scripts/shell_variables.sh 

experiment_name='fairmot_conf_038_epoch_290'
output_dir='experiments/tracking/'${experiment_name}
create_clean_directory $output_dir

python src/track.py \
    --all_external \
    --data_dir 'data/external_detections/surfrider_T1_epoch_290' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.9 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 8 \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --tracker_parameters_dir 'data/tracking_parameters' \
    --output_dir ${output_dir}


