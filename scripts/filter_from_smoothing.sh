python -m debugpy --listen 5678 --wait-for-client src/filter_tracks.py \
    --input_file experiments/tracking/EKF_1/long_segments/12fps_v0_tau_0/part_1_1.txt \
    --filter_type smoothing_v0 \
    --output_name test \
    --video_filename data/validation_videos/all/long_segments_12fps/videos/part_1_1.mp4
 