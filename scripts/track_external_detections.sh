. scripts/shell_variables.sh 

algorithm=SMC_20
details=''
experiment_name=${algorithm}${details}
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir

python src/track.py \
    --external_detections \
    --data_dir data/external_detections/FairMOT/surfrider_short_segments_12fps \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --algorithm ${algorithm} \
    --noise_covariances_path data/tracking_parameters \
    --display 0

