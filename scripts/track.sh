. scripts/shell_variables.sh 

<<<<<<< HEAD
algorithm=EKF_1
details='viz'
experiment_name=${algorithm}_${details}
=======
experiment_name=test
>>>>>>> further_research
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir


python src/track.py \
    --data_dir data/validation_videos/T1 \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --detection_threshold 0.3 \
    --downsampling_factor 4 \
    --noise_covariances_path data/tracking_parameters \
    --output_shape 960,544 \
<<<<<<< HEAD
    --skip_frames 0 \
    --display 2
=======
    --skip_frames 3 \
    --arch mobilenet_v3_small \
    --device cpu \
    --detection_batch_size 1 \
    --display 0
>>>>>>> further_research


for f in ${output_dir}/*; 
do 
    python src/postprocess_and_count_tracks.py \
        --input_file $f \
        --kappa 7  \
        --tau 4 \
        --output_name $f
    rm $f 
done
