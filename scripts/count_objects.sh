

experiment="../runs/counts/test"

python src/count_video_objects.py \
    --weights "../models/yolo_latest.pt" \
    --kappa 5 \
    --tau 3 \
    --noise_covariances_path "data/tracking_parameters" \
    --video_path "../data/videos/video_midouze15.mp4" \
    --output_dir $experiment \
    --skip_frame 5 \
    --compare 


rm -rf $experiment



