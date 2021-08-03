python src/filter_tracks.py \
    --input_file experiments/tracking/EKF_1/long_segments/12fps_v0_tau_0/part_1_1.txt \
    --filter_type smoothing_v0 \
    --output_name test \
    --frames_file data/external_detections/FairMOT/surfrider_long_segments_12fps/part_1_1/saved_frames.pickle \
    --min_len_tracklet 7 