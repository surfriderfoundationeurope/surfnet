. scripts_for_experiments/shell_variables.sh 

python build_synthetic_videos.py \
    --vid-dir ${ORIGINAL_VIDEOS} \
    --tractable-band 'center_left' \
    --output-dir ${SYNTHETIC_VIDEOS_PATH} \
    --read-every 2 \
    --original-res \
    --rescale 3 \
    --nb-extracts-per-vid 6 \
    --synthetic-objects ${SYNTHETIC_OBJECTS} \
    --nb-frames-without-object 200 \
    --max-nb-objects 3


