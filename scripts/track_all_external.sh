. scripts/shell_variables.sh 

algorithm=SMC_20
details='hungarian'
experiment_name=${algorithm}_${details}
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir

python -m debugpy --listen 5678 --wait-for-client src/track.py \
    --all_external \
    --data_dir /home/infres/chagneux/repos/FairMOT/surfrider_long_segments \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --count_threshold 0 \
    --algorithm ${algorithm} \
    --read_from folder \
    --noise_covariances_dir data/tracking_parameters

