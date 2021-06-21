. scripts/shell_variables.sh 

experiment_name='test'
output_dir='experiments/tracking/'${experiment_name}
create_clean_directory $output_dir

python src/track.py \
    --data_dir '/home/infres/chagneux/repos/FairMOT/surfrider_T1_new_data_conf_05' \
    --all_external \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 0 \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --tracker_parameters_dir 'data/tracking_parameters' \
    --output_dir ${output_dir}

