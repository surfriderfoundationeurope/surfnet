. scripts/shell_variables.sh 

experiment_name=mobilenet_v100_kappa_7_tau_4
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir


python src/track.py \
    --data_dir data/validation_videos/T1 \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --detection_threshold 0.3 \
    --downsampling_factor 4 \
    --noise_covariances_path data/tracking_parameters \
<<<<<<< HEAD
    --model_weights experiments/detection/3500_images_mobilenet/model_257.pth \
=======
    --model_weights models/mobilenet_epoch250.pth \
>>>>>>> 06b5670ae48cb0214b6d13f9b5607714df1d6136
    --output_shape 960,544 \
    --skip_frames 3 \
    --arch 'mobilenetv3small' \
    --display 0


for f in ${output_dir}/*; 
do 
    python src/postprocess_and_count_tracks.py \
        --input_file $f \
        --kappa 7  \
        --tau 4 \
        --output_name $f
    rm $f 
done
