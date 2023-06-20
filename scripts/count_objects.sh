
model_url="https://github.com/surfriderfoundationeurope/surfnet/releases/download/v01.2023/yolo_latest.pt"
wget -O models/yolo_latest.pt $model_url

experiment="../runs/counts/test"

python src/count_video_objects.py \
    --weights "models/yolo_latest.pt" \
    --kappa 5 \
    --tau 3 \
    --noise_covariances_path "data/tracking_parameters" \
    --video_path "data/validation_videos/T1/T1_1080_px_converted.mp4" \
    --output_dir $experiment \
    --skip_frame 5 


python src/count_video_objects.py \
    --weights "models/yolo_latest.pt" \
    --kappa 5 \
    --tau 3 \
    --noise_covariances_path "data/tracking_parameters" \
    --video_path "data/validation_videos/T1/T1_1080_px_converted.mp4" \
    --video_count_path "data/validation_videos_counts/T1_1080_px_converted.txt" \
    --output_dir $experiment \
    --skip_frame 5 \
    --compare

rm -rf $experiment





