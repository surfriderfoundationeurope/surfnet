. scripts_for_experiments/shell_variables.sh 

experiment_name='synthetic_videos_val_base_only'
output_dir='experiments/tracking/'${experiment_name}

create_clean_directory $output_dir 

python tracking.py \
    --base_weights ${BASE_PRETRAINED} \
    --extension_weights ${EXTENSION_PRETRAINED} \
    --data_dir ${SYNTHETIC_VIDEOS_DATASET}'data/' \
    --annotation_file 'data/synthetic_videos_dataset/annotations/annotations_val.json' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.25 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 5 \
    --detections_from_images \
    --base_only









