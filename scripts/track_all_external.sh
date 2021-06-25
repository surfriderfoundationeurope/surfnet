. scripts/shell_variables.sh 

experiment_name='fairmot_dets_T3_long_segments_conf_045_epoch_290_count_thres_8'
output_dir='experiments/tracking/'${experiment_name}
create_clean_directory $output_dir

python src/track.py \
    --all_external \
    --data_dir /home/infres/chagneux/repos/FairMOT/surfrider_T3_epoch_290 \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --count_threshold 8 \
    --algorithm 'Kalman' \
    --read_from 'folder' \
    --tracker_parameters_dir 'data/tracking_parameters' \
    --output_dir ${output_dir}


