fps=12
python scripts/run_mot_challenge.py \
    --GT_FOLDER data/gt/surfrider_part_1_only_${fps}fps \
    --TRACKERS_FOLDER data/trackers/surfrider_part_1_only_${fps}fps \
    --DO_PREPROC False \
    --USE_PARALLEL True \
    --METRICS HOTA
