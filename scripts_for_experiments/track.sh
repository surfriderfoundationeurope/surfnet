. scripts_for_experiments/shell_variables.sh 

experiment_name='synthetic_videos_val'
output_dir='experiments/tracking/'${experiment_name}

create_clean_directory $output_dir 

python -m debugpy --listen 5678 --wait-for-client tracking.py \
    --base_weights ${BASE_PRETRAINED} \
    --extension_weights ${EXTENSION_PRETRAINED} \
    --data_dir ${SYNTHETIC_VIDEOS_DATASET}'data/' \
    --annotation_file '/home/infres/chagneux/repos/surfnet/data/synthetic_videos_dataset/annotations/annotations_val.json' \
    --output_dir ${output_dir} \
    --confidence_threshold 0.33 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR}








