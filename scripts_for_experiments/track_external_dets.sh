. scripts_for_experiments/shell_variables.sh 

experiment_name='synthetic_videos_val_CenterTrack_dets'
output_dir='experiments/tracking/'${experiment_name}
external_detections_dir='data/detector_results/surfrider-test/CenterTrack'
create_clean_directory $output_dir 

python tracking.py \
    --data_dir ${SYNTHETIC_VIDEOS_DATASET}'data/' \
    --annotation_file 'data/synthetic_videos_dataset/annotations/annotations_val.json' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.2 \
    --detection_threshold 0.33 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --stop_tracking_threshold 5 \
    --external_detections_dir ${external_detections_dir}








