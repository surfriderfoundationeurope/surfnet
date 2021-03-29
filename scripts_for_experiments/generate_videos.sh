. scripts_for_experiments/shell_variables.sh 

python synthetic_videos/build_synthetic_videos.py \
    --vid-dir ${ORIGINAL_VIDEOS} \
    --tractable-band 'center_left' \
    --output-dir ${SYNTHETIC_VIDEOS_PATH} \
    --read-every 2 \
    --original-res \
    --rescale 3 \
    --nb-extracts-per-vid 20 \
    --synthetic-objects ${SYNTHETIC_OBJECTS}


