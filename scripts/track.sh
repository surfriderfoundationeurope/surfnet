. scripts/shell_variables.sh 

experiment_name=test
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir


python src/track.py \
    --data_dir data/validation_videos/T1 \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --detection_threshold 0.3 \
    --downsampling_factor 4 \
    --noise_covariances_path data/tracking_parameters \
    --model_weights models/model_289.pth \
    --output_shape 960,544 \
    --skip_frames 3 \
    --arch 'res_18' \
    --display 0


for f in ${output_dir}/*; 
do 
    python src/postprocess_and_count_tracks.py \
        --input_file $f \
        --kappa 7  \
        --tau 5 \
        --output_name $f
    rm $f 
done
