fps=12
python scripts/run_mot_challenge.py \
    --GT_FOLDER data/gt/surfrider_long_segments_${fps}fps \
    --TRACKERS_FOLDER data/trackers/surfrider_long_segments_${fps}fps \
    --DO_PREPROC False \
    --METRICS HOTA \
    --USE_PARALLEL True
