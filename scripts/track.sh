. scripts/shell_variables.sh 

experiment_name='tracking_best_synthetic_vids'
output_dir='experiments/tracking/'${experiment_name}
create_clean_directory $output_dir

python src/track.py \
    --data_dir 'data/synthetic_videos/data' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.4 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 5 \
    --algorithm 'Kalman' \
    --read_from 'annotations' \
    --annotation_file 'data/synthetic_videos/annotations/annotations_val.json' \
    --detector 'internal_base' \
    --tracker_parameters_dir 'data/tracking_parameters' \
    --base_weights 'models/BEST_MODEL_09_06.pth' \
    --output_w 960 \
    --output_h 544 \
    --skip_frames 1






