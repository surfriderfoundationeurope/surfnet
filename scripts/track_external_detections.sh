. scripts/shell_variables.sh 

algorithm=EKF_1
details='hungarian_shorts'
experiment_name=${algorithm}_${details}
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir

python src/track.py \
    --all_external \
    --data_dir /home/infres/chagneux/repos/FairMOT/surfrider_short_segments \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --algorithm ${algorithm} \
    --noise_covariances_dir data/tracking_parameters

